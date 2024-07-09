namespace PerceptivePyro.Whisper.Decoding;

using System;
using System.Collections.Generic;
using TorchSharp;

public class BeamSearchDecoder : TokenDecoder
{
    private int beamSize;
    private int eot;
    private Inference inference;
    private float patience;
    private int maxCandidates;
    private List<Dictionary<long[], float>> finishedSequences;

    public BeamSearchDecoder(int beamSize, int eot, Inference inference, float? patience = null)
    {
        this.beamSize = beamSize;
        this.eot = eot;
        this.inference = inference;
        this.patience = patience ?? 1.0f;
        this.maxCandidates = (int)Math.Round(beamSize * this.patience);
        this.finishedSequences = null;

        if (this.maxCandidates <= 0)
        {
            throw new ArgumentException($"Invalid beam size ({beamSize}) or patience ({patience})");
        }
    }

    public override void reset()
    {
        this.finishedSequences = null;
    }

    public override (Tensor tokens, bool completed) update(Tensor tokens, Tensor logits, Tensor sumLogprobs)
    {
        if (tokens.shape[0] % this.beamSize != 0)
        {
            throw new ArgumentException($"{tokens.shape}[0] % {this.beamSize} != 0");
        }

        int nAudio = (int)(tokens.shape[0] / this.beamSize);
        if (this.finishedSequences == null)  // for the first update
        {
            this.finishedSequences = Enumerable.Range(0, nAudio)
                .Select(_ => new Dictionary<long[], float>())
                .ToList();
        }

        var logprobs = torch.nn.functional.log_softmax(logits.@float(), dim: -1);
        var nextTokens = new List<long[]>();
        var sourceIndices = new List<long>();
        var finishedSequencesList = new List<Dictionary<long[], float>>();

        for (int i = 0; i < nAudio; i++)
        {
            var scores = new Dictionary<long[], float>();
            var sources = new Dictionary<long[], long>();
            var finished = new Dictionary<long[], float>();

            // STEP 1: calculate the cumulative log probabilities for possible candidates
            for (int j = 0; j < this.beamSize; j++)
            {
                int idx = i * this.beamSize + j;
                var prefix = tokens[idx].data<long>().ToList();
                var (topLogprobs, topTokens) = logprobs[idx].topk(this.beamSize + 1);
                for (int k = 0; k < topLogprobs.shape[0]; k++)
                {
                    var logprob = topLogprobs[k];
                    var token = topTokens[k];
                    var newLogprob = (sumLogprobs[idx] + logprob).item<float>();
                    var sequence = prefix.Concat([token.item<long>()]).ToArray();
                    scores[sequence] = newLogprob;
                    sources[sequence] = idx;
                }
            }

            // STEP 2: rank the candidates and keep the top beam_size sequences for each audio
            int saved = 0;
            foreach (var sequence in scores.OrderByDescending(x => x.Value).Select(x => x.Key))
            {
                if (sequence[^1] == this.eot)
                {
                    finished[sequence] = scores[sequence];
                }
                else
                {
                    sumLogprobs[nextTokens.Count] = scores[sequence];
                    nextTokens.Add(sequence);
                    sourceIndices.Add(sources[sequence]);

                    saved++;
                    if (saved == this.beamSize)
                    {
                        break;
                    }
                }
            }

            finishedSequencesList.Add(finished);
        }

        var tokensResult = torch.tensor(nextTokens.ToMultiDimensional(), device: tokens.device);
        this.inference.rearrange_kv_cache(sourceIndices.ToArray());

        // add newly finished sequences to this.finishedSequences
        if (this.finishedSequences.Count != finishedSequencesList.Count)
        {
            throw new InvalidOperationException("Number of finished sequences does not match.");
        }
        for (int i = 0; i < this.finishedSequences.Count; i++)
        {
            var previouslyFinished = this.finishedSequences[i];
            var newlyFinished = finishedSequencesList[i];
            foreach (var seq in newlyFinished.OrderByDescending(x => x.Value).Select(x => x.Key))
            {
                if (previouslyFinished.Count >= this.maxCandidates)
                {
                    break;  // the candidate list is full
                }
                previouslyFinished[seq] = newlyFinished[seq];
            }
        }

        // mark as completed if all audio has enough number of samples
        bool completed = this.finishedSequences.All(sequences => sequences.Count >= this.maxCandidates);
        return (tokensResult, completed);
    }

    public override (List<List<Tensor>> tokens, List<List<float>> sum_logprobs) finalize(Tensor tokens, Tensor sum_logprobs)
    {
        // collect all finished sequences, including patience, and add unfinished ones if not enough
        sum_logprobs = sum_logprobs.cpu();
        for (int i = 0; i < this.finishedSequences.Count; i++)
        {
            var sequences = this.finishedSequences[i];
            if (sequences.Count < this.beamSize)  // when not enough sequences are finished
            {
                foreach (int j in Enumerable.Range(0, (int)sum_logprobs.shape[1]).OrderByDescending(j => sum_logprobs[i, j]))
                {
                    var sequence = tokens[i, j].data<long>()
                        .Concat([this.eot]).ToArray();
                    sequences[sequence] = sum_logprobs[i, j].item<float>();
                    if (sequences.Count >= this.beamSize)
                    {
                        break;
                    }
                }
            }
        }

        var tokensResult = this.finishedSequences
            .Select(sequences => sequences.Keys.Select(seq => torch.tensor(seq.ToArray())).ToList())
            .ToList();
        var sumLogprobsResult = this.finishedSequences
            .Select(sequences => sequences.Values.ToList())
            .ToList();

        return (tokensResult, sumLogprobsResult);
    }
}