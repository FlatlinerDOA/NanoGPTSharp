namespace PerceptivePyro.Whisper;

using Microsoft.ML.Tokenizers;
using PerceptivePyro.Whisper.Decoding;
using System;
using System.IO;
using System.IO.Compression;
using System.Text;
using Tensorboard;
using TorchSharp;
using TorchSharp.Modules;

using CacheHookRemover = TorchSharp.torch.nn.HookableModule<Func<nn.Module<Tensor, Tensor>, Tensor, Tensor>, Func<nn.Module<Tensor, Tensor>, Tensor, Tensor, Tensor>>.HookRemover;

public class WhisperModel : nn.Module<Tensor, Tensor, Tensor>
{
    private static readonly Dictionary<string, string> _MODELS = new()
    {
        ["tiny.en"] = "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
        ["tiny"] = "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
        ["base.en"] = "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
        ["base"] = "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
        ["small.en"] = "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
        ["small"] = "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
        ["medium.en"] = "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
        ["medium"] = "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
        ["large-v1"] = "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt",
        ["large-v2"] = "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
        ["large-v3"] = "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
        ["large"] = "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
    };

    private static readonly Dictionary<string, (string Url, string Sha256)> HUGGING_FACE_MODELS = new()
    {
        ["tiny.en"] = ("openai/whisper-tiny.en", "db59695928ded6043adaef491a53ef4e12da9611184d77c53baa691a60b958ad"),
        //["tiny"] = "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
        //["base.en"] = "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
        //["base"] = "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
        //["small.en"] = "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
        //["small"] = "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
        //["medium.en"] = "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
        //["medium"] = "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
        //["large-v1"] = "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt",
        //["large-v2"] = "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
        //["large-v3"] = "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
        //["large"] = "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
    };



    private static readonly Dictionary<string, string> _ALIGNMENT_HEADS = new()
    {
        ["tiny.en"] = "ABzY8J1N>@0{>%R00Bk>$p{7v037`oCl~+#00",
        ["tiny"] = "ABzY8bu8Lr0{>%RKn9Fp%m@SkK7Kt=7ytkO",
        ["base.en"] = "ABzY8;40c<0{>%RzzG;p*o+Vo09|#PsxSZm00",
        ["base"] = "ABzY8KQ!870{>%RzyTQH3`Q^yNP!>##QT-<FaQ7m",
        ["small.en"] = "ABzY8>?_)10{>%RpeA61k&I|OI3I$65C{;;pbCHh0B{qLQ;+}v00",
        ["small"] = "ABzY8DmU6=0{>%Rpa?J`kvJ6qF(V^F86#Xh7JUGMK}P<N0000",
        ["medium.en"] = "ABzY8usPae0{>%R7<zz_OvQ{)4kMa0BMw6u5rT}kRKX;$NfYBv00*Hl@qhsU00",
        ["medium"] = "ABzY8B0Jh+0{>%R7}kK1fFL7w6%<-Pf*t^=N)Qr&0RR9",
        ["large-v1"] = "ABzY8r9j$a0{>%R7#4sLmoOs{s)o3~84-RPdcFk!JR<kSfC2yj",
        ["large-v2"] = "ABzY8zd+h!0{>%R7=D0pU<_bnWW*tkYAhobTNnu$jnkEkXqp)j;w1Tzk)UH3X%SZd&fFZ2fC2yj",
        ["large-v3"] = "ABzY8gWO1E0{>%R7(9S+Kn!D~%ngiGaR?*L!iJG9p-nab0JQ=-{D1-g00",
        ["large"] = "ABzY8gWO1E0{>%R7(9S+Kn!D~%ngiGaR?*L!iJG9p-nab0JQ=-{D1-g00",
    };

    public ModelDimensions dims;
    public AudioEncoder encoder;
    public TextDecoder decoder;
    public Tensor all_heads;

    public WhisperModel(ModelDimensions dims) : base(nameof(WhisperModel))
    {
        this.dims = dims;
        this.encoder = new AudioEncoder(
        this.dims.n_mels,
        this.dims.n_audio_ctx,
        this.dims.n_audio_state,
        this.dims.n_audio_head,
        this.dims.n_audio_layer
    );
        this.decoder = new TextDecoder(
            this.dims.n_vocab,
            this.dims.n_text_ctx,
            this.dims.n_text_state,
            this.dims.n_text_head,
            this.dims.n_text_layer
        );

        // use the last half among the decoder layers for time alignment by public object ault;
        // to use a specific set of heads, see `set_alignment_heads()` below.
        this.all_heads = torch.zeros(this.dims.n_text_layer, this.dims.n_text_head, dtype: torch.@bool);
        this.all_heads[(this.dims.n_text_layer / 2)..] = true;
        this.register_buffer("alignment_heads", this.all_heads.to_sparse(), persistent: false);
    }

    public static byte[] Base85Decode(string input) => Encoding.ASCII.GetBytes(input)
        .Select(x => (byte)(x - 33))
        .ToArray();

    public static byte[] Decompress(byte[] data)
    {
        using (var compressedStream = new MemoryStream(data))
        using (var decompressedStream = new MemoryStream())
        using (var gzipStream = new GZipStream(compressedStream, CompressionMode.Decompress))
        {
            gzipStream.CopyTo(decompressedStream);
            return decompressedStream.ToArray();
        }
    }

    private void set_alignment_heads(string dump)
    {
        // Decode the base85 string
        var decodedBytes = Base85Decode(dump);

        // Decompress the gzip data
        var decompressedBytes = Decompress(decodedBytes);

        // Convert the byte array to a boolean array
        bool[] boolArray = decompressedBytes.Select(b => b != 0).ToArray();

        // Convert the boolean array to a Torch tensor
        var array = torch.tensor(boolArray, dtype: torch.@bool);
        var mask = array.reshape(this.dims.n_text_layer, this.dims.n_text_head);
        this.register_buffer("alignment_heads", mask.to_sparse(), persistent: false);
    }

    public Tensor embed_audio(Tensor mel) => this.encoder.call(mel);

    public Tensor logits(Tensor tokens, Tensor audio_features) => this.decoder.call((tokens, audio_features, null));

    public override Tensor forward(Tensor mel, Tensor tokens) => this.decoder.call((tokens, this.encoder.call(mel), null));

    public Device device => this.parameters().First().device;

    public bool is_multilingual => this.dims.n_vocab >= 51865;

    public int num_languages => this.dims.n_vocab - 51765 - (this.is_multilingual ? 1 : 0);

    /// <summary>
    /// Load a Whisper ASR model
    /// </summary>
    /// <param name="name">one of the official model names listed by `whisper.available_models()`, or path to a model checkpoint containing the model dimensions and the model state_dict.</param>
    /// <param name="device">the PyTorch device to put the model into.</param>
    /// <param name="download_root">path to download the model files; by default, it uses "~/.cache/whisper".</param>
    /// <param name="in_memory">whether to preload the model weights into host memory.</param>
    /// <returns>The Whisper ASR model instance.</returns>
    public static async Task<WhisperModel> LoadModelAsync(string name, Device? device = default, string? download_root = null, bool in_memory = false)
    {
        /* if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if download_root is None:
        default = os.path.join(os.path.expanduser("~"), ".cache")
        download_root = os.path.join(os.getenv("XDG_CACHE_HOME", default), "whisper")

    if name in _MODELS:
        checkpoint_file = _download(_MODELS[name], download_root, in_memory)
        alignment_heads = _ALIGNMENT_HEADS[name]
    elif os.path.isfile(name):
        checkpoint_file = open(name, "rb").read() if in_memory else name
        alignment_heads = None
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}"
        )

    with (
        io.BytesIO(checkpoint_file) if in_memory else open(checkpoint_file, "rb")
    ) as fp:
        checkpoint = torch.load(fp, map_location=device)
    del checkpoint_file

    dims = ModelDimensions(**checkpoint["dims"])
    model = Whisper(dims)
    model.load_state_dict(checkpoint["model_state_dict"])

    if alignment_heads is not None:
        model.set_alignment_heads(alignment_heads)

    return model.to(device)*/

        if (device == null)
        {
            device = torch.cuda.is_available() ? new Device(DeviceType.CUDA) : new Device(DeviceType.CPU);
        }

        if (download_root == null)
        {
            string defaultPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache");
            download_root = Path.Combine(Environment.GetEnvironmentVariable("XDG_CACHE_HOME") ?? defaultPath, "whisper");
        }

        Stream checkpointFile;
        ModelDimensions dims;
        string? alignmentHeads = null;

        if (HUGGING_FACE_MODELS.ContainsKey(name))
        {
            var (modelName, checksum) = HUGGING_FACE_MODELS[name];
            // TODO: checksum
            var path = await SafeTensors.DownloadWeightsAsync(modelName);
            dims = new ModelDimensions(30, 80, 160, 1, 1, 1, 1, 1, 1, 1);
            var state_dict = SafeTensors.LoadFile(path, device).ToDictionary(k => k.Name, k => k.Tensor);
            var model = new WhisperModel(dims);
            model.load_state_dict(state_dict);
            return model.to(device);

        }
        else if (_MODELS.ContainsKey(name))
        {
            checkpointFile = await DownloadAsync(_MODELS[name], download_root, in_memory);
            alignmentHeads = _ALIGNMENT_HEADS.GetValueOrDefault(name);
        }
        else if (File.Exists(name))
        {
            checkpointFile = File.OpenRead(name);
            alignmentHeads = null;
        }
        else
        {
            throw new NotSupportedException($"Model {name} not found; available models = {string.Join(", ", _MODELS.Keys)}");
        }

        using (checkpointFile)
        {
            throw new NotImplementedException();

            //var checkpoint = torch.load(checkpointFile, device);
            //// TODO: checkpoint["dims"].ToObject<Dictionary<string, int>>();
            //var model = new WhisperModel(dims);
            //model.load_state_dict(checkpoint["model_state_dict"].ToTensor());

            //if (alignmentHeads is not null)
            //{
            //    model.SetAlignmentHeads(alignmentHeads.Value);
            //}

            //return model.to(device);
        }
    }

    private static async Task<Stream> DownloadAsync(string url, string download_root, bool in_memory)
    {
        var filePath = Path.Combine(download_root, url[(url.LastIndexOf('/') + 1)..]);
        Directory.CreateDirectory(Path.GetDirectoryName(filePath)!);
        $"Downloading weights from {url} to {filePath}".Dump();
        var stream = await new HttpClient().GetStreamAsync(url);


        if (in_memory)
        {
            var ms = new MemoryStream();
            await stream.CopyToAsync(ms);
            ms.Position = 0;
            return ms;
        }

        using (var outputStream = File.OpenWrite(filePath))
        {
            await stream.CopyToAsync(outputStream);
        }

        return File.OpenRead(filePath);
    }

    public (Dictionary<nn.Module, Tensor> cache, List<CacheHookRemover> hooks) install_kv_cache_hooks(Dictionary<nn.Module, Tensor> cache = null)
    {
        /*
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        */
        cache = cache is not null ? new(cache) : new();
        var hooks = new List<CacheHookRemover>();

        Tensor save_to_cache(nn.Module<Tensor, Tensor> module, Tensor _, Tensor output)
        {
            if (!cache.ContainsKey(module) || output.shape[1] > this.dims.n_text_ctx)
            {
                // save as-is, for the first token or cross attention
                cache[module] = output;
            }
            else
            {
                cache[module] = torch.cat([cache[module], output], dim: 1).detach();
            }
            return cache[module];
        }

        void install_hooks(nn.Module layer)
        {
            if (layer is MultiHeadAttention head)
            {
                hooks.Add(head.key.register_forward_hook(save_to_cache));
                hooks.Add(head.value.register_forward_hook(save_to_cache));
            }

        }

        this.decoder.apply(install_hooks);
        return (cache, hooks);
    }

    public (Tensor lang_tokens, List<Dictionary<string, float>> lang_probs) detect_language(Tensor audio_features, WhisperTokenizer tokenizer) => throw new NotImplementedException();  //Decoding.detect_language_function();
    
    public TranscriptionResult transcribe(string path) => throw new NotImplementedException(); // Decoding.transcribe_function();


    public DecodingResult decode(Tensor mel, DecodingOptions? options = null, Dictionary<string, object>? kwargs = null) => decode_batch(mel, options, kwargs).First();

    public IEnumerable<DecodingResult> decode_batch(Tensor mel, DecodingOptions? options = null, Dictionary<string, object>? kwargs = null)
    {
        using var _ = no_grad();

        var single = mel.ndim == 2;
        if (single)
        {
            mel = mel.unsqueeze(0);
        }

        if (kwargs is not null)
        {
            options = options.With(kwargs);
        }

        var result = new DecodingTask(this, options).Run(mel);
        return single ? result.Take(1) : result;
    }
}
