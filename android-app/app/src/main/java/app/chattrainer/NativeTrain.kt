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
import androidx.compose.material3.Checkbox
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.util.zip.ZipInputStream

/**
 * Trains a LoRA adapter on the phone by running Tether's `llama-finetune-lora`
 * (the QVAC fork of llama.cpp) on a small GGUF model.
 *
 * The trainer and the model are downloaded once, into the app's private storage.
 * The URLs are editable in the UI because release locations can move.
 */
object NativeTrainer {
    const val DEFAULT_TRAINER_URL =
        "https://github.com/tetherto/qvac-fabric/releases/download/v1.0/qvac-android-adreno-arm64-v1.0.zip"
    const val DEFAULT_MODEL_URL =
        "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf"

    @Volatile
    var process: Process? = null

    fun dir(ctx: Context) = File(ctx.filesDir, "native").apply { mkdirs() }
    fun modelFile(ctx: Context) = File(dir(ctx), "model.gguf")
    fun adapterFile(ctx: Context) = File(File(ctx.filesDir, "output").apply { mkdirs() }, "phone_adapter.gguf")

    fun findBinary(ctx: Context): File? =
        File(dir(ctx), "trainer").walkTopDown().firstOrNull { it.isFile && it.name == "llama-finetune-lora" }

    fun download(url: String, dest: File, onProgress: (String) -> Unit) {
        val tmp = File(dest.path + ".part")
        val c = URL(url.trim()).openConnection() as HttpURLConnection
        c.connectTimeout = 15000
        c.readTimeout = 30000
        c.instanceFollowRedirects = true
        try {
            if (c.responseCode !in 200..299) throw RuntimeException("Download failed: HTTP ${c.responseCode} for $url")
            val total = c.contentLengthLong
            var done = 0L
            var lastShown = 0L
            c.inputStream.use { input ->
                tmp.outputStream().use { out ->
                    val buf = ByteArray(256 * 1024)
                    while (true) {
                        val n = input.read(buf)
                        if (n < 0) break
                        out.write(buf, 0, n)
                        done += n
                        if (done - lastShown > 4_000_000) {
                            lastShown = done
                            onProgress(
                                "Downloading ${dest.name}: ${done / 1_000_000} MB" +
                                    if (total > 0) " of ${total / 1_000_000} MB" else "",
                            )
                        }
                    }
                }
            }
            if (dest.exists()) dest.delete()
            if (!tmp.renameTo(dest)) throw RuntimeException("Could not save ${dest.name}")
        } finally {
            c.disconnect()
            tmp.delete()
        }
    }

    fun unzip(zip: File, into: File) {
        into.deleteRecursively()
        into.mkdirs()
        val root = into.canonicalPath + File.separator
        ZipInputStream(zip.inputStream().buffered()).use { z ->
            while (true) {
                val e = z.nextEntry ?: break
                val out = File(into, e.name)
                if (!out.canonicalPath.startsWith(root)) continue // zip-slip guard
                if (e.isDirectory) {
                    out.mkdirs()
                } else {
                    out.parentFile?.mkdirs()
                    out.outputStream().use { z.copyTo(it) }
                }
            }
        }
        into.walkTopDown().filter { it.isFile }.forEach { it.setExecutable(true, false) }
    }

    /** Runs training; calls [onLine] for every line of output. Returns the exit code. */
    fun train(
        ctx: Context, epochs: Int, ctxLen: Int, useGpu: Boolean, onLine: (String) -> Unit,
    ): Int {
        val bin = findBinary(ctx) ?: throw RuntimeException("Trainer not installed yet. Press Download trainer + model.")
        val data = File(File(ctx.filesDir, "output"), "train.jsonl")
        if (!data.exists()) throw RuntimeException("Build the dataset first (sections 1-3).")
        val model = modelFile(ctx)
        if (!model.exists()) throw RuntimeException("Model missing. Press Download trainer + model.")
        val trainerRoot = File(dir(ctx), "trainer")
        val libDirs = listOfNotNull(
            bin.parentFile, bin.parentFile?.parentFile, trainerRoot,
        ).map { it.absolutePath }.distinct().joinToString(":")
        val cmd = listOf(
            bin.absolutePath, "-m", model.absolutePath, "-f", data.absolutePath,
            "--assistant-loss-only", "-c", ctxLen.toString(), "-b", "128", "-ub", "128",
            "-ngl", if (useGpu) "99" else "0", "-fa", "off",
            "--num-epochs", epochs.toString(),
            "--output-adapter", adapterFile(ctx).absolutePath,
        )
        val pb = ProcessBuilder(cmd).redirectErrorStream(true).directory(dir(ctx))
        pb.environment()["LD_LIBRARY_PATH"] = libDirs
        val p = try {
            pb.start()
        } catch (e: java.io.IOException) {
            throw RuntimeException(
                "This phone refused to run the trainer (${e.message}). Android blocks running downloaded programs " +
                    "on some versions. Use the PC option (section 4) instead.",
            )
        }
        process = p
        p.inputStream.bufferedReader().forEachLine(onLine)
        return p.waitFor().also { process = null }
    }
}

@Composable
fun NativePanel(datasetReady: Boolean) {
    val ctx = LocalContext.current
    val view = LocalView.current
    val scope = rememberCoroutineScope()
    var trainerUrl by rememberSaveable { mutableStateOf(NativeTrainer.DEFAULT_TRAINER_URL) }
    var modelUrl by rememberSaveable { mutableStateOf(NativeTrainer.DEFAULT_MODEL_URL) }
    var epochs by rememberSaveable { mutableStateOf("1") }
    var useGpu by rememberSaveable { mutableStateOf(true) }
    var info by remember { mutableStateOf("") }
    var log by remember { mutableStateOf("") }
    var busy by remember { mutableStateOf(false) }

    // keep the screen awake while a download or a training run is in progress
    DisposableEffect(busy) {
        view.keepScreenOn = busy
        onDispose { view.keepScreenOn = false }
    }

    fun run(block: () -> Unit) {
        busy = true
        scope.launch {
            try {
                withContext(Dispatchers.IO) { block() }
            } catch (e: Exception) {
                info = "Error: ${e.message ?: e.javaClass.simpleName}"
            }
            busy = false
        }
    }

    Text("5. Train on this phone (small models)", style = MaterialTheme.typography.titleMedium)
    Text(
        "Runs Tether's QVAC LoRA trainer on a small model (Qwen3 0.6B). Expect hours, a warm phone and a drained battery. " +
            "Keep the app open and the phone plugged in. Big models (7-8B) cannot be trained on a phone: use section 4.",
        style = MaterialTheme.typography.bodySmall,
    )
    OutlinedTextField(
        value = trainerUrl, onValueChange = { trainerUrl = it }, label = { Text("Trainer zip URL") },
        modifier = Modifier.fillMaxWidth(),
    )
    OutlinedTextField(
        value = modelUrl, onValueChange = { modelUrl = it }, label = { Text("Model .gguf URL") },
        modifier = Modifier.fillMaxWidth(),
    )
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
        OutlinedTextField(
            value = epochs, onValueChange = { epochs = it.filter(Char::isDigit).take(2) },
            label = { Text("Epochs") }, singleLine = true, modifier = Modifier.weight(1f),
        )
        Checkbox(checked = useGpu, onCheckedChange = { useGpu = it })
        Text("Use phone GPU (Vulkan)", modifier = Modifier.weight(2f))
    }
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        OutlinedButton(enabled = !busy, onClick = {
            run {
                val d = NativeTrainer.dir(ctx)
                val zip = File(d, "trainer.zip")
                NativeTrainer.download(trainerUrl, zip) { info = it }
                info = "Unpacking trainer…"
                NativeTrainer.unzip(zip, File(d, "trainer"))
                zip.delete()
                if (NativeTrainer.findBinary(ctx) == null) {
                    throw RuntimeException("That zip has no llama-finetune-lora inside. Check the trainer URL.")
                }
                if (!NativeTrainer.modelFile(ctx).exists()) {
                    NativeTrainer.download(modelUrl, NativeTrainer.modelFile(ctx)) { info = it }
                }
                info = "Trainer and model are ready."
            }
        }) { Text("Download trainer + model") }
        Button(enabled = !busy && datasetReady, onClick = {
            run {
                log = ""
                info = "Training… (this can take hours)"
                val code = NativeTrainer.train(ctx, epochs.toIntOrNull()?.coerceAtLeast(1) ?: 1, 256, useGpu) { line ->
                    log = (log + line + "\n").takeLast(6000)
                }
                info = if (code == 0 && NativeTrainer.adapterFile(ctx).exists()) {
                    "Training finished. Adapter: ${NativeTrainer.adapterFile(ctx).length() / 1024} KB."
                } else {
                    "Training ended with code $code. See the log."
                }
            }
        }) { Text("Start") }
        OutlinedButton(enabled = busy, onClick = { NativeTrainer.process?.destroy() }) { Text("Stop") }
    }
    if (NativeTrainer.adapterFile(ctx).exists()) {
        OutlinedButton(onClick = {
            val uri = FileProvider.getUriForFile(ctx, ctx.packageName + ".files", NativeTrainer.adapterFile(ctx))
            val send = Intent(Intent.ACTION_SEND).apply {
                type = "application/octet-stream"
                putExtra(Intent.EXTRA_STREAM, uri)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            ctx.startActivity(Intent.createChooser(send, "Save adapter"))
        }) { Text("Share adapter (.gguf)") }
    }
    if (info.isNotEmpty()) Text(info, style = MaterialTheme.typography.bodyMedium)
    if (log.isNotEmpty()) {
        Column(Modifier.fillMaxWidth().heightIn(max = 240.dp).verticalScroll(rememberScrollState())) {
            Text(log, style = MaterialTheme.typography.bodySmall)
        }
    }
}
