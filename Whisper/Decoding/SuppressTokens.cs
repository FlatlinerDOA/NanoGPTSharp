namespace PerceptivePyro.Whisper.Decoding;

public class SuppressTokens : LogitFilter
{
    private readonly TensorIndex suppress_tokens;

    public SuppressTokens(IEnumerable<int> suppress_tokens)
    {
        this.suppress_tokens = TensorIndex.Tensor(suppress_tokens.ToArray());
    }

    public override void apply(Tensor logits, Tensor tokens)
    {
        logits[.., this.suppress_tokens] = float.NegativeInfinity;
    }
}
