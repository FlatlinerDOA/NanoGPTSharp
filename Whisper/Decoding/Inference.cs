namespace PerceptivePyro.Whisper.Decoding;

public abstract class Inference
{
    public abstract Tensor logits(Tensor tokens, Tensor audioFeatures);
    public abstract void rearrange_kv_cache(long[] sourceIndices);
    public virtual void cleanup_caching() { }
}
