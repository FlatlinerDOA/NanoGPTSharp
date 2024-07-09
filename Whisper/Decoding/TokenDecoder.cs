namespace PerceptivePyro.Whisper.Decoding;

public abstract class TokenDecoder
{
    /// <summary>
    /// Initialize any stateful variables for decoding a new sequence
    /// </summary>
    public virtual void reset()
    {
    }

    /// <summary>
    /// Specify how to select the next token, based on the current trace and logits
    /// </summary>
    /// <param name="tokens">Tensor, shape = (n_batch, current_sequence_length) -  all tokens in the context so far, including the prefix and sot_sequence tokens.</param>
    /// <param name="logits">Tensor, shape = (n_batch, vocab_size) - per-token logits of the probability distribution at the current step.</param>
    /// <param name="sum_logprobs">sum_logprobs : Tensor, shape = (n_batch) - cumulative log probabilities for each sequence.</param>
    /// <returns>
    /// Tensor tokens - shape = (n_batch, current_sequence_length + 1) - the tokens, appended with the selected next token,
    /// bool completed - True if all sequences has reached the end of text.
    /// </returns>
    public abstract (Tensor tokens, bool completed) update(Tensor tokens, Tensor logits, Tensor sum_logprobs);

    /// <summary>
    /// Finalize search and return the final candidate sequences
    /// </summary>
    /// <param name="tokens"> tokens : Tensor, shape = (n_audio, n_group, current_sequence_length) - all tokens in the context so far, including the prefix and sot_sequence.</param>
    /// <param name="sum_logprobs">sum_logprobs : Tensor, shape = (n_audio, n_group) - cumulative log probabilities for each sequence.</param>
    /// <returns>
    /// tokens : Sequence[Sequence[Tensor]], length = n_audio - sequence of Tensors containing candidate token sequences, for each audio input.
    /// sum_logprobs : List[List[float]], length = n_audio - sequence of cumulative log probabilities corresponding to the above.
    /// </returns>
    public abstract (List<List<Tensor>> tokens, List<List<float>> sum_logprobs) finalize(Tensor tokens, Tensor sum_logprobs);
}
