namespace PerceptivePyro.GPT;

using System;
using System.Diagnostics;
using System.Diagnostics.Contracts;
using PerceptivePyro;
using TorchSharp.Modules;
using F = nn.functional;

public class GPTModel : nn.Module<Tensor, Tensor?, bool, (Tensor logits, Tensor? loss)>
{
    private GPTConfig config;
    private ModuleDict<nn.Module> transformer;
    private Linear lm_head;

    public GPTModel(GPTConfig config) : base(nameof(GPTModel))
    {
        Contract.Assert(config.vocab_size is not 0);
        Contract.Assert(config.block_size is not 0);

        this.config = config;

        // Word Token Embeddings
        var wte = nn.Embedding(config.vocab_size, config.n_embd);

        // Word Positional Embeddings
        var wpe = nn.Embedding(config.block_size, config.n_embd);

        var drop = nn.Dropout(config.dropout);

        var ln_f = new LayerNorm(config.n_embd, hasBias: config.has_bias);

        // Transformer Layers
        var h = nn.ModuleList((from _ in Enumerable.Range(0, config.n_layer) select new TransformerBlock(config)).ToArray());
        transformer = nn.ModuleDict(
            ("wte", wte),
            ("wpe", wpe),
            ("drop", drop),
            ("h", h),
            ("ln_f", ln_f)
        );

        lm_head = nn.Linear(config.n_embd, config.vocab_size, hasBias: false);

        // with weight tying when using torch.compile() some warnings get generated:
        // "UserWarning: functional_call was passed multiple values for tied weights.
        // This behavior is deprecated and will be an error in future versions"
        // not 100% sure what this is, so far seems to be harmless.
        // TODO: investigate
        this.wte.weight = lm_head.weight; // https://paperswithcode.com/method/weight-tying

        // init all weights
        apply(_init_weights);
        // apply special scaled init to the residual projections, per GPT-2 paper
        foreach (var (pn, p) in named_parameters())
        {
            if (pn.EndsWith("c_proj.weight"))
            {
                nn.init.normal_(p, mean: 0.0, std: 0.02 / Math.Sqrt(2 * config.n_layer));
            }
        }

        RegisterComponents();

        // report number of parameters
        Debug.WriteLine($"number of parameters: {get_num_params() / 1e6}M");
    }

    private Embedding wte => (Embedding)transformer["wte"];

    private Embedding wpe => (Embedding)transformer["wpe"];

    private Dropout drop => (Dropout)transformer["drop"];

    private LayerNorm ln_f => (LayerNorm)transformer["ln_f"];
    private ModuleList<TransformerBlock> h => (ModuleList<TransformerBlock>)transformer["h"];

    public static async Task<GPTModel> from_pretrained(string model_type, string device, GPTConfig override_args = null)
    {
        // n_layer, n_head and n_embd are determined from model_type
        var model_configs = new Dictionary<string, GPTConfig>()
        {
            ["gpt2"] = new() { n_layer = 12, n_head = 12, n_embd = 768 },         // 124M params
            ["gpt2-medium"] = new() { n_layer = 24, n_head = 16, n_embd = 1024 }, // 350M params
            ["gpt2-large"] = new() { n_layer = 36, n_head = 20, n_embd = 1280, },  // 774M params
            ["gpt2-xl"] = new() { n_layer = 48, n_head = 25, n_embd = 1600 },     // 1558M params
        };

        Contract.Assert(model_configs.ContainsKey(model_type), $"Invalid model_type: {model_type}");
        var config = model_configs[model_type];

        // forcing vocab_size=50257, block_size=1024, bias=True
        config = config with
        {
            vocab_size = 50257, // always 50257 for GPT model checkpoints
            block_size = 1024, // always 1024 for GPT model checkpoints
            has_bias = true, // always True for GPT model checkpoints,
            dropout = override_args?.dropout ?? config.dropout, // overriding dropout rate
        };

        // create a from-scratch initialized minGPT model
        var model = new GPTModel(config);
        model.to(device);
        var sd = model.state_dict();
        var sd_keys = from kv in sd where !kv.Key.EndsWith(".attn.bias") select kv; // discard this mask / buffer, not a param

        // init a huggingface/transformers model
        var safeTensorsFilePath = await SafeTensors.DownloadWeightsAsync(model_type);
        var sd_hf = (from t in SafeTensors.LoadFile(safeTensorsFilePath, device)
                     where !t.Name.EndsWith(".attn.masked_bias") && !t.Name.EndsWith(".attn.bias") // ignore these, just a buffer
                     select new KeyValuePair<string, Tensor>(t.Name, t.Tensor)).ToDictionary(k => k.Key, k => k.Value);

        // basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        // this means that we have to transpose these weights when we import them.
        var transposed = new[] { "attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight" };

        // copy while ensuring all of the parameters are aligned and match in names and shapes (NOTE: In the .safetensors lm_head.weight is not stored as it is the same as wte.weight (they are tied)
        Contract.Assert(sd_hf.Keys.Count() + 1 == sd_keys.Count(), $"mismatched keys: {sd_hf.Keys.Count() + 1} != {sd_keys.Count()}");

        // Stream tensors into the the dictionary.
        foreach (var (name, target) in sd_keys)
        {
            // HACK: Translate names.
            var translated_name = name.StartsWith("transformer.") ? name.Substring("transformer.".Length) :
                name == "lm_head.weight" ? "wte.weight" :
                name;
            var source = sd_hf.GetValueOrDefault(translated_name);
            Contract.Assert(source is not null, $"Could not find matching tensor for {name} ({translated_name})");
            if (transposed.Any(name.EndsWith))
            {
                // special treatment for the Conv1D weights we need to transpose
                Contract.Assert(source.shape.Reverse().SequenceEqual(target.shape), "Shape's not right");
                using (var _ = no_grad())
                {
                    target.copy_(source.t());
                    //$"Transposed {name} {target.shape.Stringify()}".Dump();
                    //target.Dump();
                }
            }
            else
            {
                // vanilla copy over the other parameters
                Contract.Assert(source.shape.SequenceEqual(target.shape), $"Size of loaded tensor {name}: ({source.shape.Stringify()}) does not match configured ({target.shape.Stringify()})");
                using (var _ = no_grad())
                {
                    target.copy_(source);
                    //$"Loaded {name} {target.shape.Stringify()}".Dump();
                    //target.Dump();
                }
            }
        }

        // Default to inference mode for most use cases.
        model.eval();
        return model;
    }

    public override (Tensor, Tensor?) forward(Tensor idx, Tensor? targets = null, bool embeddings_only = false)
    {
        var device = idx.device;
        var (b, t) = (idx.size()[0], idx.size()[1]);
        Contract.Assert(t <= config.block_size, $"Cannot forward sequence of length {t}, block size is only {config.block_size}");
        var pos = arange(0, t, dtype: @long, device: device).unsqueeze(0); // shape (1, t)

        // forward the GPT model itself
        var tok_emb = wte.call(idx); // token embeddings of shape (b, t, n_embd)
        var pos_emb = wpe.call(pos); // position embeddings of shape (1, t, n_embd)
        var x = drop.call(tok_emb + pos_emb);
        foreach (var block in h)
        {
            x = block.call(x);
        }

        x = ln_f.call(x);

        if (embeddings_only)
        {
            return (x, null);
        }

        Tensor logits;
        Tensor? loss;
        if (targets is not null)
        {
            //if we are given some desired targets also calculate the loss
            logits = lm_head.call(x);
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index: -1);
        }
        else
        {
            // inference-time mini-optimization: only forward the lm_head on the very last position
            long lastIndex = x.shape[1] - 1;
            var x_narrowed = x.narrow(1, lastIndex, 1); // Equivalent to x[:, [-1], :]
            logits = lm_head.forward(x_narrowed); // note: using list [-1] to preserve the time dim
            loss = null;
        }


        return (logits, loss);
    }

    /// <summary>
    /// Return the number of parameters in the model.
    /// For non-embedding count(default), the position embeddings get subtracted.
    /// The token embeddings would too, except due to the parameter sharing these
    /// params are actually used as weights in the final layer, so we include them.
    /// </summary>
    /// <param name="non_embedding"></param>
    /// <returns></returns>
    public long get_num_params(bool non_embedding = true)
    {
        var n_params = (from p in parameters()
                        select p.numel()).Sum();
        if (non_embedding)
        {
            n_params -= wpe.weight?.numel() ?? 0;
        }

        return n_params;
    }

    /// <summary>
    /// model surgery to decrease the block size if necessary
    /// e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
    /// but want to use a smaller block size for some smaller, simpler model
    /// </summary>
    /// <param name="new_block_size">New desired block size</param>
    /// <returns></returns>
    public void crop_block_size(int new_block_size)
    {
        Contract.Assert(new_block_size <= config.block_size);
        config = config with { block_size = new_block_size };
        wpe.weight = nn.Parameter(wpe.weight[..new_block_size]);
        foreach (var block in h)
        {
            var bias = block.attn.bias;
            if (bias is not null)
            {
                block.attn.bias = bias[.., .., ..new_block_size, ..new_block_size];
            }
        }
    }

    public Tensor generate_embedding(Tensor context)
    {
        using var no_grad = torch.no_grad();

        // if the sequence context is growing too long we must crop it at block_size
        using var idx_cond = context.size(1) <= config.block_size ? context : context[.., -config.block_size..];

        // forward the model to get the logits for the index in the sequence
        var (logits, _) = call(idx_cond, null, true);
        return logits; // (B*, Last(T), C*)
    }

    /// <summary>
    /// Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    /// the sequence max_new_tokens times, feeding the predictions back into the model each time.
    /// Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    /// </summary>
    /// <param name="idx"></param>
    /// <param name="max_new_tokens"></param>
    /// <param name="temperature"></param>
    /// <param name="top_k"></param>
    /// <returns></returns>
    public Tensor generate(Tensor idx, int max_new_tokens, double temperature = 1.0d, int? top_k = null)
    {
        using var no_grad = torch.no_grad();
        foreach (var token in Enumerable.Range(0, max_new_tokens))
        {
            $"Step {token}".Dump();

            // if the sequence context is growing too long we must crop it at block_size
            using var idx_cond = idx.size(1) <= config.block_size ? idx : idx[.., -config.block_size..];

            // forward the model to get the logits for the index in the sequence
            var (logits, _) = call(idx_cond, null, false);

            // pluck the logits at the final step and scale by desired temperature
            logits = logits[.., -1, ..] / temperature;

            // optionally crop the logits to only the top k options
            if (top_k is int k)
            {
                var top = Math.Min(k, (int)logits.size(-1));
                var (v, top_indexes) = topk(logits, top);
                var lowest = v[.., -1].reshape(1, 1);
                ////lowest.Dump();
                logits.masked_fill_(logits < lowest, float.NegativeInfinity);
                ////logits.ToString(TorchSharp.TensorStringStyle.Julia, "0.00", 520000).Dump();
            }

            // apply softmax to convert logits to (normalized) probabilities
            using var probs = F.softmax(logits, dim: -1);

            // sample from the distribution
            using var idx_next = multinomial(probs, num_samples: 1);

            // append sampled index to the running sequence and continue
            idx = cat(new[] { idx, idx_next }, dim: 1);
        }

        return idx;
    }

    private void _init_weights(nn.Module module)
    {
        if (module is Linear lin)
        {
            nn.init.normal_(lin.weight, mean: 0.0, std: 0.02);
            if (lin.bias is not null)
            {
                nn.init.zeros_(lin.bias);
            }
        }
        else if (module is Embedding e)
        {
            nn.init.normal_(e.weight, mean: 0.0, std: 0.02);
        }
    }
}
