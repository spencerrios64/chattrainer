package app.chattrainer

/** One chat message. [ts] is unix seconds, 0 when unknown. [isMe] null means "decide from the names list". */
data class Message(
    val platform: String,
    val chatId: String,
    val sender: String,
    var text: String,
    val ts: Double = 0.0,
    val isMe: Boolean? = null,
)

data class Options(
    val name: String = "me",
    val names: List<String> = emptyList(),
    val scrubPii: Boolean = true,
    val scrubUrls: Boolean = true,
    val anonymizeNames: Boolean = false,
    val skipGroups: Boolean = false,
    val mode: String = "both", // pairs | windows | both
    val contextTurns: Int = 6,
    val windowTurns: Int = 12,
    val windowStride: Int = 6,
    val mergeGapMinutes: Double = 3.0,
    val sessionGapHours: Double = 8.0,
    val minReplyChars: Int = 2,
    val maxTurnChars: Int = 1200,
    val includeSolo: Boolean = true,
    val valFraction: Double = 0.03,
    val seed: Int = 42,
) {
    fun system(): String =
        "You are $name. Reply exactly the way $name texts: same tone, slang, language, humor and message length."

    fun myNames(): Set<String> = names.map { it.lowercase() }.toSet()
}
