package app.chattrainer

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Checkbox
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface(Modifier.fillMaxSize()) { App() }
            }
        }
    }
}

private fun displayName(ctx: Context, uri: Uri): String {
    ctx.contentResolver.query(uri, null, null, null, null)?.use { c ->
        val i = c.getColumnIndex(OpenableColumns.DISPLAY_NAME)
        if (i >= 0 && c.moveToFirst()) return c.getString(i)
    }
    return uri.lastPathSegment ?: "file"
}

/** Read the picked exports, build train/val JSONL into app storage, return a summary. */
private fun runBuild(ctx: Context, files: List<Pair<Uri, String>>, o: Options): String {
    val chats = LinkedHashMap<String, MutableList<Message>>()
    for ((uri, fileName) in files) {
        val text = ctx.contentResolver.openInputStream(uri)?.use { it.readBytes().toString(Charsets.UTF_8) } ?: continue
        for (m in Parsers.parse(fileName, text)) chats.getOrPut(m.chatId) { mutableListOf() }.add(m)
    }
    if (chats.isEmpty()) {
        return "No messages found. Add a WhatsApp .txt, Telegram result.json or Instagram message_1.json file."
    }
    val r = DatasetBuilder.build(chats, o)
    val dir = File(ctx.filesDir, "output").apply { mkdirs() }
    File(dir, "train.jsonl").writeText(r.train.joinToString("\n", postfix = if (r.train.isEmpty()) "" else "\n"))
    File(dir, "val.jsonl").writeText(r.validation.joinToString("\n", postfix = if (r.validation.isEmpty()) "" else "\n"))
    val lines = r.stats.entries.joinToString("\n") { "${it.key}: ${it.value}" }
    val warn = if (r.stats.keys.none { it.endsWith("_my_messages") && (r.stats[it] ?: 0) > 0 }) {
        "\n\nNone of the messages matched your names. Add your name exactly as it appears in the chat."
    } else ""
    return lines + warn
}

private fun share(ctx: Context) {
    val dir = File(ctx.filesDir, "output")
    val uris = arrayListOf<Uri>()
    for (n in listOf("train.jsonl", "val.jsonl")) {
        val f = File(dir, n)
        if (f.exists()) uris.add(FileProvider.getUriForFile(ctx, ctx.packageName + ".files", f))
    }
    if (uris.isEmpty()) return
    val send = Intent(Intent.ACTION_SEND_MULTIPLE).apply {
        type = "*/*"
        putParcelableArrayListExtra(Intent.EXTRA_STREAM, uris)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    ctx.startActivity(Intent.createChooser(send, "Send dataset"))
}

@Composable
private fun CheckRow(label: String, value: Boolean, onChange: (Boolean) -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Checkbox(checked = value, onCheckedChange = onChange)
        Text(label)
    }
}

@Composable
fun App() {
    val ctx = LocalContext.current
    val scope = rememberCoroutineScope()
    var name by rememberSaveable { mutableStateOf("me") }
    var names by rememberSaveable { mutableStateOf("") }
    var scrubPii by rememberSaveable { mutableStateOf(true) }
    var scrubUrls by rememberSaveable { mutableStateOf(true) }
    var anonymize by rememberSaveable { mutableStateOf(false) }
    var skipGroups by rememberSaveable { mutableStateOf(false) }
    var mode by rememberSaveable { mutableStateOf("both") }
    val files = remember { mutableStateListOf<Pair<Uri, String>>() }
    var status by remember { mutableStateOf("") }
    var busy by remember { mutableStateOf(false) }
    var built by remember { mutableStateOf(false) }

    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
        uris.forEach { u -> files.add(u to displayName(ctx, u)) }
    }

    Column(
        Modifier
            .fillMaxSize()
            .statusBarsPadding()
            .navigationBarsPadding()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text("chattrainer", style = MaterialTheme.typography.headlineMedium)
        Text(
            "Turn your own chat exports into a training dataset on your phone. " +
                "Train on your PC over Wi-Fi (section 4) or on the phone itself (section 5, small models, slow).",
            style = MaterialTheme.typography.bodyMedium,
        )

        Text("1. Your chat exports", style = MaterialTheme.typography.titleMedium)
        Text("WhatsApp .txt, Telegram result.json, Instagram message_1.json", style = MaterialTheme.typography.bodySmall)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = { picker.launch(arrayOf("*/*")) }) { Text("Add files") }
            OutlinedButton(onClick = { files.clear() }, enabled = files.isNotEmpty()) { Text("Clear") }
        }
        files.forEach { Text("• ${it.second}", style = MaterialTheme.typography.bodySmall) }

        Text("2. Who are you?", style = MaterialTheme.typography.titleMedium)
        OutlinedTextField(
            value = name, onValueChange = { name = it }, label = { Text("Your name") },
            singleLine = true, modifier = Modifier.fillMaxWidth(),
        )
        OutlinedTextField(
            value = names, onValueChange = { names = it },
            label = { Text("Names that are YOU in the chats (comma separated)") },
            modifier = Modifier.fillMaxWidth(),
        )

        Text("3. Privacy and format", style = MaterialTheme.typography.titleMedium)
        CheckRow("Scrub emails, phone numbers, IBANs, card-like numbers", scrubPii) { scrubPii = it }
        CheckRow("Replace URLs", scrubUrls) { scrubUrls = it }
        CheckRow("Anonymize other people's names", anonymize) { anonymize = it }
        CheckRow("Only 1-on-1 chats (skip groups)", skipGroups) { skipGroups = it }
        Row(verticalAlignment = Alignment.CenterVertically) {
            listOf("pairs", "windows", "both").forEach { m ->
                RadioButton(selected = mode == m, onClick = { mode = m })
                Text(m, modifier = Modifier.padding(end = 12.dp))
            }
        }

        Button(
            enabled = !busy && files.isNotEmpty(),
            onClick = {
                busy = true
                built = false
                status = "Building…"
                val opts = Options(
                    name = name.trim().ifEmpty { "me" },
                    names = names.split(",").map { it.trim() }.filter { it.isNotEmpty() },
                    scrubPii = scrubPii, scrubUrls = scrubUrls,
                    anonymizeNames = anonymize, skipGroups = skipGroups, mode = mode,
                )
                val picked = files.toList()
                scope.launch {
                    status = try {
                        withContext(Dispatchers.Default) { runBuild(ctx, picked, opts) }.also { built = true }
                    } catch (e: Exception) {
                        "Something went wrong: ${e.message}"
                    }
                    busy = false
                }
            },
        ) { Text("Build dataset") }

        if (status.isNotEmpty()) Text(status, style = MaterialTheme.typography.bodyMedium)
        if (built) {
            Button(onClick = { share(ctx) }) { Text("Send train.jsonl to my PC") }
            Text(
                "On the PC: copy train.jsonl and val.jsonl into the chattrainer output folder, " +
                    "then press Start training on the Train tab (skip Build dataset).",
                style = MaterialTheme.typography.bodySmall,
            )
        }
        val ready = built || File(File(ctx.filesDir, "output"), "train.jsonl").exists()
        RemotePanel(datasetReady = ready)
        NativePanel(datasetReady = ready)
    }
}
