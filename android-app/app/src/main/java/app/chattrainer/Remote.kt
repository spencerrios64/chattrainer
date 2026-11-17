package app.chattrainer

import android.content.Context
import android.content.Intent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

/** Talks to `chattrainer serve` running on the PC (same Wi-Fi). */
class RemoteClient(address: String, private val code: String) {
    private val base = "http://" + address.trim().removePrefix("http://").trimEnd('/')

    private fun call(
        method: String, path: String, body: ByteArray? = null, timeoutMs: Int = 15000,
    ): ByteArray {
        val c = URL(base + path).openConnection() as HttpURLConnection
        try {
            c.requestMethod = method
            c.connectTimeout = 6000
            c.readTimeout = timeoutMs
            c.setRequestProperty("X-Token", code.trim())
            if (body != null) {
                c.doOutput = true
                c.setFixedLengthStreamingMode(body.size)
                c.outputStream.use { it.write(body) }
            }
            val ok = c.responseCode in 200..299
            val stream = if (ok) c.inputStream else c.errorStream
            val bytes = stream?.use { it.readBytes() } ?: ByteArray(0)
            if (!ok) {
                val msg = runCatching { JSONObject(String(bytes)).optString("error") }.getOrDefault("")
                throw RuntimeException(msg.ifEmpty { "HTTP ${c.responseCode}" })
            }
            return bytes
        } finally {
            c.disconnect()
        }
    }

    fun status(): JSONObject = JSONObject(String(call("GET", "/api/status")))
    fun sendDataset(name: String, data: ByteArray) {
        call("POST", "/api/dataset?name=$name", data, timeoutMs = 120000)
    }

    fun start(model: String) {
        call("POST", "/api/train", JSONObject().put("model", model).toString().toByteArray())
    }

    fun stop() {
        call("POST", "/api/stop", ByteArray(0))
    }

    fun log(from: Int): String = String(call("GET", "/api/log?from=$from"), Charsets.UTF_8)
    fun adapter(): ByteArray = call("GET", "/api/adapter", timeoutMs = 120000)
}

private fun shareFile(ctx: Context, f: File) {
    val uri = FileProvider.getUriForFile(ctx, ctx.packageName + ".files", f)
    val send = Intent(Intent.ACTION_SEND).apply {
        type = "application/zip"
        putExtra(Intent.EXTRA_STREAM, uri)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    ctx.startActivity(Intent.createChooser(send, "Save adapter"))
}

@Composable
fun RemotePanel(datasetReady: Boolean) {
    val ctx = LocalContext.current
    val scope = rememberCoroutineScope()
    val prefs = remember { ctx.getSharedPreferences("remote", Context.MODE_PRIVATE) }
    var address by rememberSaveable { mutableStateOf(prefs.getString("address", "") ?: "") }
    var code by rememberSaveable { mutableStateOf(prefs.getString("code", "") ?: "") }
    var model by rememberSaveable { mutableStateOf("") }
    var info by remember { mutableStateOf("Not connected.") }
    var log by remember { mutableStateOf("") }
    var polling by remember { mutableStateOf(false) }

    fun client() = RemoteClient(address, code)
    fun guarded(block: suspend () -> Unit) {
        prefs.edit().putString("address", address).putString("code", code).apply()
        scope.launch {
            try {
                withContext(Dispatchers.IO) { block() }
            } catch (e: Exception) {
                info = "Error: ${e.message ?: e.javaClass.simpleName}"
            }
        }
    }

    LaunchedEffect(polling) {
        var offset = 0
        while (polling) {
            try {
                val (txt, st) = withContext(Dispatchers.IO) {
                    val c = client()
                    c.log(offset) to c.status()
                }
                if (txt.isNotEmpty()) {
                    offset += txt.toByteArray(Charsets.UTF_8).size
                    log = (log + txt).takeLast(6000)
                }
                if (!st.optBoolean("busy")) {
                    info = if (st.optInt("exit_code", 1) == 0) "Training finished." else "Training stopped or failed. See log."
                    polling = false
                }
            } catch (e: Exception) {
                info = "Lost connection: ${e.message}"
                polling = false
            }
            delay(2000)
        }
    }

    Text("4. Train on my PC", style = MaterialTheme.typography.titleMedium)
    Text(
        "On the PC open the chattrainer app, Train tab, press Phone remote. It shows an address and a code. " +
            "Phone and PC must be on the same Wi-Fi.",
        style = MaterialTheme.typography.bodySmall,
    )
    OutlinedTextField(
        value = address, onValueChange = { address = it }, label = { Text("PC address, e.g. 192.168.1.20:8765") },
        singleLine = true, modifier = Modifier.fillMaxWidth(),
    )
    OutlinedTextField(
        value = code, onValueChange = { code = it }, label = { Text("Pairing code") },
        singleLine = true, modifier = Modifier.fillMaxWidth(),
    )
    OutlinedTextField(
        value = model, onValueChange = { model = it },
        label = { Text("Model (blank = what the PC already uses)") },
        singleLine = true, modifier = Modifier.fillMaxWidth(),
    )
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        OutlinedButton(onClick = {
            guarded {
                val s = client().status()
                info = "Connected. Model: ${s.optString("model")}, " +
                    "GPU memory: ${if (s.isNull("vram_gb")) "none found" else "${s.optDouble("vram_gb").toInt()} GB"}, " +
                    "dataset on PC: ${s.optBoolean("has_dataset")}, trained adapter: ${s.optBoolean("has_adapter")}"
            }
        }) { Text("Connect") }
        Button(enabled = datasetReady, onClick = {
            guarded {
                info = "Sending dataset…"
                val dir = File(ctx.filesDir, "output")
                val c = client()
                c.sendDataset("train", File(dir, "train.jsonl").readBytes())
                val v = File(dir, "val.jsonl")
                if (v.exists()) c.sendDataset("val", v.readBytes())
                info = "Dataset sent to the PC."
            }
        }) { Text("Send dataset") }
    }
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Button(onClick = {
            guarded {
                client().start(model.trim())
                log = ""
                info = "Training started on the PC."
                polling = true
            }
        }) { Text("Start training") }
        OutlinedButton(onClick = { guarded { client().stop(); info = "Stop requested." } }) { Text("Stop") }
        OutlinedButton(onClick = {
            guarded {
                info = "Downloading adapter…"
                val f = File(File(ctx.filesDir, "output").apply { mkdirs() }, "adapter.zip")
                f.writeBytes(client().adapter())
                info = "Adapter downloaded (${f.length() / 1024} KB)."
                withContext(Dispatchers.Main) { shareFile(ctx, f) }
            }
        }) { Text("Get adapter") }
    }
    Text(info, style = MaterialTheme.typography.bodyMedium)
    if (log.isNotEmpty()) {
        Column(Modifier.fillMaxWidth().heightIn(max = 240.dp).verticalScroll(rememberScrollState())) {
            Text(log, style = MaterialTheme.typography.bodySmall)
        }
    }
}
