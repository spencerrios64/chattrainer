package app.chattrainer

import org.json.JSONArray
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.charset.CharacterCodingException
import java.nio.charset.CodingErrorAction
import java.time.LocalDateTime
import java.time.ZoneOffset

/** Export parsers: WhatsApp (.txt), Telegram (result.json) and Instagram (message_*.json). */
object Parsers {
    /** Pick a parser from the file name and parse its text. Unknown files give an empty list. */
    fun parse(fileName: String, rawText: String): List<Message> {
        val text = rawText.removePrefix("﻿")
        val lower = fileName.lowercase()
        return try {
            when {
                lower.endsWith(".txt") -> whatsapp(fileName.substringBeforeLast('.'), text)
                lower.endsWith(".json") && lower.startsWith("message_") -> instagram(fileName, text)
                lower.endsWith(".json") -> telegram(fileName.substringBeforeLast('.'), text)
                else -> emptyList()
            }
        } catch (e: Exception) {
            emptyList()
        }
    }

    // ------------------------------------------------------------ WhatsApp
    private val WA_LINE = Regex(
        """^[‎‏]?\[?(?<date>\d{1,4}[./-]\d{1,2}[./-]\d{1,4}),?\s+""" +
            """(?<time>\d{1,2}[:.]\d{2}(?:[:.]\d{2})?(?:\s?[APap]\.?[Mm]\.?)?)\]?\s*(?:-\s*)?""" +
            """(?<sender>[^:]{1,80}?):\s(?<text>.*)$"""
    )
    private val WA_SYSTEM = listOf(
        "<media omitted>", "<attached:", "image omitted", "video omitted", "sticker omitted",
        "audio omitted", "this message was deleted", "you deleted this message", "<this message was edited>",
    )

    private fun waTs(date: String, time: String): Double {
        return try {
            val d = date.split(Regex("[./-]"))
            var t = time.trim().uppercase().replace(Regex("\\s"), "")
            t = t.replace("A.M.", "AM").replace("P.M.", "PM").replace(".", ":")
            val pm = t.endsWith("PM")
            val am = t.endsWith("AM")
            t = t.removeSuffix("PM").removeSuffix("AM")
            val tp = t.split(":").mapNotNull { it.toIntOrNull() }
            if (tp.size < 2) return 0.0
            var h = tp[0]
            if (pm && h < 12) h += 12
            if (am && h == 12) h = 0
            val a = d[0].toInt()
            val b = d[1].toInt()
            val c = d[2].toInt()
            val year: Int
            val month: Int
            val day: Int
            if (d[0].length == 4) {
                year = a; month = b; day = c
            } else {
                year = if (c < 100) 2000 + c else c
                if (a <= 31 && b <= 12) { day = a; month = b } else { month = a; day = b }
            }
            LocalDateTime.of(year, month, day, h, tp[1], tp.getOrElse(2) { 0 })
                .toEpochSecond(ZoneOffset.UTC).toDouble()
        } catch (e: Exception) {
            0.0
        }
    }

    fun whatsapp(stem: String, text: String): List<Message> {
        val chatId = "whatsapp:$stem"
        val msgs = mutableListOf<Message>()
        for (raw in text.lines()) {
            val m = WA_LINE.matchEntire(raw)
            if (m != null) {
                val g = m.groups
                msgs.add(
                    Message(
                        "whatsapp", chatId, g["sender"]!!.value.trim(), g["text"]!!.value,
                        waTs(g["date"]!!.value, g["time"]!!.value),
                    )
                )
            } else if (msgs.isNotEmpty() && raw.isNotBlank()) {
                msgs.last().text += "\n" + raw // continuation of a multi-line message
            }
        }
        return msgs.filter { x -> WA_SYSTEM.none { h -> x.text.lowercase().contains(h) } }
    }

    // ------------------------------------------------------------ Telegram
    private fun flat(v: Any?): String = when (v) {
        is String -> v
        is JSONArray -> buildString {
            for (i in 0 until v.length()) {
                when (val p = v.get(i)) {
                    is String -> append(p)
                    is JSONObject -> append(p.optString("text", ""))
                }
            }
        }
        else -> ""
    }

    private fun telegramChat(chat: JSONObject, fallback: String): List<Message> {
        val label = chat.optString("name").ifEmpty { chat.opt("id")?.toString() ?: fallback }
        val chatId = "telegram:$label"
        val out = mutableListOf<Message>()
        val arr = chat.optJSONArray("messages") ?: return out
        for (i in 0 until arr.length()) {
            val m = arr.optJSONObject(i) ?: continue
            if (m.optString("type") != "message") continue
            val text = flat(m.opt("text")).trim()
            if (text.isEmpty()) continue
            val ts = try {
                LocalDateTime.parse(m.getString("date")).toEpochSecond(ZoneOffset.UTC).toDouble()
            } catch (e: Exception) {
                0.0
            }
            out.add(Message("telegram", chatId, m.optString("from").ifEmpty { "?" }, text, ts))
        }
        return out
    }

    fun telegram(stem: String, text: String): List<Message> {
        val data = JSONObject(text)
        val chats = data.optJSONObject("chats")
        if (chats != null) {
            val list = chats.optJSONArray("list") ?: return emptyList()
            return (0 until list.length()).flatMap { i ->
                telegramChat(list.getJSONObject(i), i.toString())
            }
        }
        return if (data.has("messages")) telegramChat(data, stem) else emptyList()
    }

    // ----------------------------------------------------------- Instagram
    /** Instagram/Facebook exports store UTF-8 bytes as latin-1 code points. */
    fun fixMojibake(s: String): String {
        if (s.any { it.code > 255 }) return s
        return try {
            Charsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT)
                .decode(ByteBuffer.wrap(s.toByteArray(Charsets.ISO_8859_1)))
                .toString()
        } catch (e: CharacterCodingException) {
            s
        }
    }

    fun instagram(fileName: String, text: String): List<Message> {
        val data = JSONObject(text)
        val arr = data.optJSONArray("messages") ?: return emptyList()
        val chatId = "instagram:" + fixMojibake(data.optString("title", fileName))
        val out = mutableListOf<Message>()
        for (i in 0 until arr.length()) {
            val m = arr.optJSONObject(i) ?: continue
            val raw = m.optString("content", "")
            if (raw.isEmpty()) continue
            val content = fixMojibake(raw)
            if (content.endsWith("to your message ") || content.contains(" sent an attachment")) continue
            out.add(
                Message(
                    "instagram", chatId, fixMojibake(m.optString("sender_name", "?")), content,
                    m.optLong("timestamp_ms", 0L) / 1000.0,
                )
            )
        }
        return out.sortedBy { it.ts } // Meta exports are newest-first
    }
}
