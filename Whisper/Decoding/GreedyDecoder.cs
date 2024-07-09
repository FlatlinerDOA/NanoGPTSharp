namespace PerceptivePyro.Whisper.Decoding;

using System.Collections.Generic;
using TorchSharp.Modules;
using F = nn.functional;

public class GreedyDecoder : TokenDecoder
{
    private float temperature;
    private int eot;

    public GreedyDecoder(float temperature, int eot)
    {
        this.temperature = temperature;
        this.eot = eot;
    }

    /// <inheritdoc/>
    public override (Tensor tokens, bool completed) update(Tensor tokens, Tensor logits, Tensor sum_logprobs)
    {
        var next_tokens = (this.temperature == 0) ?
            logits.argmax(dim: -1) :
            new Categorical(logits = logits / this.temperature).sample();

        var logprobs = F.log_softmax(logits.@float(), dim: -1);
        var current_logprobs = logprobs[torch.arange(logprobs.shape[0]), next_tokens];
        sum_logprobs += current_logprobs * (tokens[.., -1] != this.eot);

        next_tokens[tokens[.. ^1].eq(this.eot)] = this.eot;
        tokens = torch.cat([tokens, next_tokens[.., TensorIndex.None]], dim: -1);

        var completed = (tokens[.., -1].eq(this.eot)).all().item<bool>();
        return (tokens, completed);
    }

    /// <inheritdoc/>
    public override (List<List<Tensor>> tokens, List<List<float>> sum_logprobs) finalize(Tensor tokens, Tensor sum_logprobs)
    {
        // make sure each sequence has at least one EOT token at the end
        tokens = F.pad(tokens, (0, 1), value: this.eot);

        var output_tokens = new List<List<Tensor>>();
        for (int i = 0; i < tokens.shape[0]; i++)
        {
            var list = new List<Tensor>();
            for (int j = 0; j < tokens.shape[1]; j++)
            {
                list[i] = tokens[i, j, ..];
            }

            output_tokens.Add(list);
        }        

        return (output_tokens, sum_logprobs.ToLists<float>().ToList());
    }
}