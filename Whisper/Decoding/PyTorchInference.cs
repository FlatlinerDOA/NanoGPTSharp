namespace PerceptivePyro.Whisper.Decoding;

using System;
using System.Collections.Generic;
using System.Linq;

using CacheHookRemover = TorchSharp.torch.nn.HookableModule<Func<TorchSharp.torch.nn.Module<Tensor, Tensor>, Tensor, Tensor>, Func<nn.Module<Tensor, Tensor>, Tensor, Tensor, Tensor>>.HookRemover;

public class PyTorchInference : Inference
{
    private WhisperModel model;
    private int initialTokenLength;
    private Dictionary<nn.Module, Tensor> kvCache;
    private List<CacheHookRemover> hooks;
    private List<Linear> kvModules;

    public PyTorchInference(WhisperModel model, int initialTokenLength)
    {
        this.model = model;
        this.initialTokenLength = initialTokenLength;
        this.kvCache = new Dictionary<nn.Module, Tensor>();
        this.hooks = new List<CacheHookRemover>();

        var keyModules = model.decoder.Blocks.Select(block => block.Attn.key).ToList();
        var valueModules = model.decoder.Blocks.Select(block => block.Attn.value).ToList();
        this.kvModules = keyModules.Concat(valueModules).ToList();
    }

    public override Tensor logits(Tensor tokens, Tensor audioFeatures)
    {
        if (!kvCache.Any())
        {
            (this.kvCache, this.hooks) = model.install_kv_cache_hooks();
        }

        if (tokens.shape[-1] > initialTokenLength)
        {
            // only need to use the last token except in the first forward pass
            tokens = tokens[TensorIndex.Ellipsis, TensorIndex.Slice(-1, null, null)];
        }

        return model.decoder.call((tokens, audioFeatures, kvCache));
    }

    public override void cleanup_caching()
    {
        foreach (var hook in hooks)
        {
            hook.remove();
        }

        kvCache.Clear();
        hooks.Clear();
    }

    public override void rearrange_kv_cache(long[] sourceIndices)
    {
        if (!sourceIndices.SequenceEqual(Enumerable.Range(0, sourceIndices.Length).Cast<long>()))
        {
            foreach (var module in kvModules)
            {
                // update the key/value cache to contain the selected sequences
                kvCache[module] = kvCache[module].index_select(0, sourceIndices).detach();
            }
        }
    }
}
