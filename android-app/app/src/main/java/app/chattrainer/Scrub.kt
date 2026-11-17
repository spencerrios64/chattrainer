package app.chattrainer

/** PII scrubbing applied to training text (same rules as the desktop app). */
object Scrub {
    private val EMAIL = Regex("""[\w.+-]+@[\w-]+(?:\.[\w-]+)+""")
    private val URL = Regex("""(?:https?://|www\.)\S+""", RegexOption.IGNORE_CASE)
    private val IBAN = Regex("""\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}(?:\s?[A-Z0-9]{1,4})?\b""")
    private val CARD = Regex("""\b(?:\d[ -]?){13,19}\b""")
    private val PHONE = Regex("""(?<!\w)(?:\+|00)?\d[\d\s().-]{7,}\d(?!\w)""")

    fun scrub(input: String, urls: Boolean = true, names: List<String>? = null): String {
        var text = EMAIL.replace(input, "<EMAIL>")
        if (urls) text = URL.replace(text, "<URL>")
        text = IBAN.replace(text, "<IBAN>")
        text = CARD.replace(text, "<NUMBER>")
        text = PHONE.replace(text, "<PHONE>")
        for (n in names.orEmpty()) {
            if (n.length >= 3) {
                text = Regex(Regex.escape(n), RegexOption.IGNORE_CASE).replace(text, "<NAME>")
            }
        }
        return text
    }
}
