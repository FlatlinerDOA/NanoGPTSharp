using Microsoft.ML.Tokenizers;

namespace PerceptivePyro.Whisper.Decoding;

public class SuppressBlank : LogitFilter 
{
    private WhisperTokenizer tokenizer;
    private int sample_begin;

    public SuppressBlank(WhisperTokenizer tokenizer, int sample_begin)
    {
        this.tokenizer = tokenizer;
        this.sample_begin = sample_begin;

    }

    public override void apply(Tensor logits, Tensor tokens)
    {
        if (tokens.shape[1] == this.sample_begin)
        {
            var ids = this.tokenizer.Encode(" ")
                .Append(this.tokenizer.eot)
                .Select(i => (long)i)
                .ToArray();
            logits[.., TensorIndex.Tensor(ids)] = float.NegativeInfinity;
        }
    }
}
