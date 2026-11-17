package app.chattrainer

import org.json.JSONArray
import org.json.JSONObject
import java.math.BigInteger
import java.security.MessageDigest
import kotlin.math.max
import kotlin.math.min

private typealias Example = List<Pair<String, String>>

/** Turns parsed messages into chat-format training examples (JSONL), like the desktop app. */
object DatasetBuilder {
    private class Turn(val role: String, var text: String)

    class BuildResult(val train: List<String>, val validation: List<String>, val stats: Map<String, Int>)

    private fun isMe(m: Message, o: Options): Boolean = m.isMe ?: (m.sender.lowercase() in o.myNames())

    /** Merge consecutive same-speaker messages and split on long gaps. */
    private fun turns(input: List<Message>, o: Options): List<List<Turn>> {
        var msgs = input
        if (msgs.all { it.ts > 0 }) msgs = msgs.sortedBy { it.ts }
        val group = msgs.filter { !isMe(it, o) }.map { it.sender }.toSet().size > 1
        val names = if (o.anonymizeNames) msgs.map { it.sender }.filter { it != "__me__" }.toSortedSet().toList()
        else emptyList()
        val sessions = mutableListOf<List<Turn>>()
        var cur = mutableListOf<Turn>()
        var last: Message? = null
        for (m in msgs) {
            var text = m.text.trim().take(o.maxTurnChars)
            if (o.scrubPii) text = Scrub.scrub(text, o.scrubUrls, names)
            if (text.isEmpty()) continue
            val me = isMe(m, o)
            val role = if (me) "assistant" else "user"
            if (!me && group) {
                val who = if (o.scrubPii) Scrub.scrub(m.sender, false) else m.sender
                text = if (o.anonymizeNames) "<NAME>: $text" else "$who: $text"
            }
            val prev = last
            val gap = if (prev != null && m.ts > 0 && prev.ts > 0) m.ts - prev.ts else 0.0
            if (prev != null && gap > o.sessionGapHours * 3600) {
                if (cur.isNotEmpty()) sessions.add(cur)
                cur = mutableListOf()
            }
            if (cur.isNotEmpty() && cur.last().role == role && (gap == 0.0 || gap <= o.mergeGapMinutes * 60)) {
                cur.last().text += "\n" + text
            } else {
                cur.add(Turn(role, text))
            }
            last = m
        }
        if (cur.isNotEmpty()) sessions.add(cur)
        return sessions
    }

    private fun examples(session: List<Turn>, o: Options): List<Example> {
        val sys = "system" to o.system()
        val out = mutableListOf<Example>()
        if (o.mode == "pairs" || o.mode == "both") {
            for ((i, t) in session.withIndex()) {
                if (t.role != "assistant" || t.text.length < o.minReplyChars || i == 0) continue
                var ctx = session.subList(max(0, i - o.contextTurns), i).toList()
                while (ctx.isNotEmpty() && ctx[0].role != "user") ctx = ctx.drop(1)
                if (ctx.isNotEmpty()) {
                    out.add(listOf(sys) + ctx.map { it.role to it.text } + ("assistant" to t.text))
                }
            }
        }
        if ((o.mode == "windows" || o.mode == "both") && session.size >= 2) {
            var s = 0
            while (s < session.size) {
                var w = session.subList(s, min(session.size, s + o.windowTurns)).toList()
                while (w.isNotEmpty() && w.first().role != "user") w = w.drop(1)
                while (w.isNotEmpty() && w.last().role != "assistant") w = w.dropLast(1)
                if (w.size >= 2) out.add(listOf(sys) + w.map { it.role to it.text })
                s += max(1, o.windowStride)
            }
        }
        return out
    }

    /** Chats that contain only your side (e.g. Snapchat) become standalone replies. */
    private fun solo(session: List<Turn>, o: Options): List<Example> {
        val sys = "system" to o.system()
        return session.filter { it.role == "assistant" && it.text.length >= o.minReplyChars }
            .map { listOf(sys, "user" to "(keep chatting)", "assistant" to it.text) }
    }

    private fun toJson(ex: Example): String {
        val arr = JSONArray()
        for ((r, c) in ex) arr.put(JSONObject().put("role", r).put("content", c))
        return JSONObject().put("messages", arr).toString()
    }

    fun build(chats: Map<String, List<Message>>, o: Options): BuildResult {
        val train = mutableListOf<Example>()
        val val_ = mutableListOf<Example>()
        val stats = linkedMapOf<String, Int>()
        fun bump(k: String, n: Int = 1) { stats[k] = (stats[k] ?: 0) + n }
        for ((chatId, msgs) in chats) {
            val platform = msgs.first().platform
            val senders = msgs.filter { !isMe(it, o) }.map { it.sender }.toSet()
            if (o.skipGroups && senders.size > 1) {
                bump("groups_skipped")
                continue
            }
            val mine = msgs.count { isMe(it, o) }
            bump("${platform}_messages", msgs.size)
            bump("${platform}_my_messages", mine)
            if (mine == 0) {
                bump("chats_without_me")
                continue
            }
            val solo = senders.isEmpty()
            if (solo && !o.includeSolo) continue
            val exs = turns(msgs, o).flatMap { if (solo) solo(it, o) else examples(it, o) }
            // keep every chat entirely in train or val so conversations never leak across the split
            val h = BigInteger(1, MessageDigest.getInstance("MD5").digest(chatId.toByteArray()))
                .mod(BigInteger.valueOf(1000)).toInt() / 1000.0
            (if (h < o.valFraction) val_ else train).addAll(exs)
        }
        val shuffled = train.shuffled(kotlin.random.Random(o.seed))
        stats["chats"] = chats.size
        stats["train_examples"] = shuffled.size
        stats["val_examples"] = val_.size
        return BuildResult(shuffled.map(::toJson), val_.map(::toJson), stats)
    }
}
