namespace PerceptivePyro.Whisper.Decoding;

public abstract class LogitFilter
{
    /// <summary>
    /// Apply any filtering or masking to logits in-place
    /// </summary>
    /// <param name="logits">Tensor, shape = (n_batch, vocab_size) - per-token logits of the probability distribution at the current step.</param>
    /// <param name="tokens">Tensor, shape = (n_batch, current_sequence_length) - all tokens in the context so far, including the prefix and sot_sequence tokens.</param>
    public abstract void apply(Tensor logits, Tensor tokens);
}
