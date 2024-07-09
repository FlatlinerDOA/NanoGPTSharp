namespace PerceptivePyro.Whisper.Decoding;

using Google.Protobuf.WellKnownTypes;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO.Compression;
using System.Linq;
using System.Text;


public class DecodingTask
{
    private Inference inference;
    private SequenceRanker sequence_ranker;
    private TokenDecoder decoder;
    private List<LogitFilter> logit_filters;

    private WhisperModel model;
    private WhisperTokenizer tokenizer;
    private DecodingOptions options;
    private int n_group;
    private int n_ctx;
    private int sample_len;
    private int[] sot_sequence;
    private int[] initial_tokens;
    private int sample_begin;
    private int sot_index;

    public DecodingTask(WhisperModel model, DecodingOptions options)
    {
        this.model = model;

        string language = options.language ?? "en";

        this.tokenizer = WhisperTokenizer.GetTokenizer(model.is_multilingual, model.num_languages, language, options.task);
        this.options = VerifyOptions(options);

        this.n_group = options.beam_size ?? options.best_of ?? 1;
        this.n_ctx = model.dims.n_text_ctx;
        this.sample_len = options.sample_len ?? model.dims.n_text_ctx / 2;

        this.sot_sequence = options.without_timestamps ? tokenizer.sot_sequence_including_notimestamps : tokenizer.sot_sequence;
        this.initial_tokens = GetInitialTokens();
        this.sample_begin = this.initial_tokens.Length;
        this.sot_index = Array.IndexOf(this.initial_tokens, tokenizer.sot);

        this.inference = new PyTorchInference(model, this.initial_tokens.Length);
        this.sequence_ranker = new MaximumLikelihoodRanker(options.length_penalty);

        if (options.beam_size.HasValue)
        {
            this.decoder = new BeamSearchDecoder(options.beam_size.Value, tokenizer.eot, this.inference, options.patience);
        }
        else
        {
            this.decoder = new GreedyDecoder(options.temperature, tokenizer.eot);
        }

        this.logit_filters = new List<LogitFilter>();
        if (this.options.suppress_blank)
        {
            this.logit_filters.Add(new SuppressBlank(this.tokenizer, this.sample_begin));
        }

        if (this.options.suppress_tokens != null)
        {
            this.logit_filters.Add(new SuppressTokens(GetSuppressTokens()));
        }

        if (!options.without_timestamps)
        {
            throw new NotImplementedException();
            //double precision = CHUNK_LENGTH / model.dims.n_audio_ctx;
            //int? max_initial_timestamp_index = options.max_initial_timestamp.HasValue ?
            //    (int?)(Math.Round(options.max_initial_timestamp.Value / precision)) : null;

            //this.logit_filters.Add(new ApplyTimestampRules(tokenizer, this.sample_begin, max_initial_timestamp_index));
        }
    }

    private DecodingOptions VerifyOptions(DecodingOptions options)
    {
        if (options.beam_size.HasValue && options.best_of.HasValue)
        {
            throw new ArgumentException("beam_size and best_of can't be given together");
        }
        if (options.temperature == 0)
        {
            if (options.best_of.HasValue)
            {
                throw new ArgumentException("best_of with greedy sampling (T=0) is not compatible");
            }
        }
        if (options.patience.HasValue && !options.beam_size.HasValue)
        {
            throw new ArgumentException("patience requires beam_size to be given");
        }
        if (options.length_penalty.HasValue && !(0 <= options.length_penalty.Value && options.length_penalty.Value <= 1))
        {
            throw new ArgumentException("length_penalty (alpha) should be a value between 0 and 1");
        }

        return options;
    }

    private int[] GetInitialTokens()
    {
        List<int> tokens = new List<int>(this.sot_sequence);

        if (!string.IsNullOrEmpty(this.options.prefix.text))
        {
            var prefix_tokens = this.tokenizer.Encode(" " + this.options.prefix.text.Trim());
            if (this.sample_len != 0)
            {
                int max_prefix_len = this.n_ctx / 2 - this.sample_len;
                prefix_tokens = prefix_tokens.Skip(Math.Max(0, prefix_tokens.Count - max_prefix_len)).ToArray();
            }
            tokens.AddRange(prefix_tokens);
        }

        if (!string.IsNullOrEmpty(this.options.prompt.text))
        {
            var prompt_tokens = this.tokenizer.Encode(" " + this.options.prompt.text.Trim());
            tokens = new List<int> { this.tokenizer.sot_prev }
                .Concat(prompt_tokens.Skip(Math.Max(0, prompt_tokens.Count - (this.n_ctx / 2 - 1))))
                .Concat(tokens).ToList();
        }

        return tokens.ToArray();
    }

    private int[] GetSuppressTokens()
    {
        int[] suppress_tokens = this.options.suppress_tokens?.Split(',').Select(int.Parse).ToArray();

        if (suppress_tokens.Contains(-1))
        {
            suppress_tokens = suppress_tokens.Where(t => t >= 0).Concat(this.tokenizer.non_speech_tokens).ToArray();
        }
        else if (suppress_tokens == null || suppress_tokens.Length == 0)
        {
            suppress_tokens = new int[] { };
        }

        suppress_tokens = suppress_tokens.Concat(new int[]
        {
                this.tokenizer.transcribe,
                this.tokenizer.translate,
                this.tokenizer.sot,
                this.tokenizer.sot_prev,
                this.tokenizer.sot_lm,
        }).ToArray();

        if (this.tokenizer.no_speech.HasValue)
        {
            suppress_tokens = suppress_tokens.Append(this.tokenizer.no_speech.Value).ToArray();
        }

        return suppress_tokens.Distinct().OrderBy(t => t).ToArray();
    }

    private Tensor GetAudioFeatures(Tensor mel)
    {
        if (this.options.fp16)
        {
            mel = mel.half();
        }

        Tensor audio_features;
        if (mel.shape[-2] == this.model.dims.n_audio_ctx && mel.shape[-1] == this.model.dims.n_audio_state)
        {
            audio_features = mel;
        }
        else
        {
            audio_features = this.model.encoder.forward(mel);
        }

        if (audio_features.dtype != (this.options.fp16 ? torch.float16 : torch.float32))
        {
            throw new InvalidOperationException($"audio_features has an incorrect dtype: {audio_features.dtype}");
        }

        return audio_features;
    }

    private (List<string> languages, List<Dictionary<string, float>> lang_probs) DetectLanguage(Tensor audio_features, Tensor tokens)
    {
        List<string> languages = Enumerable.Repeat(this.options.language, (int)audio_features.shape[0]).ToList();
        List<Dictionary<string, float>> lang_probs = null;

        if (this.options.language == null || this.options.task == "lang_id")
        {
            (Tensor lang_tokens, List<Dictionary<string, float>> lang_probs) detectionResult = this.model.detect_language(audio_features, this.tokenizer);
            lang_probs = detectionResult.lang_probs;
            languages = lang_probs.Select(probs => probs.OrderByDescending(kv => kv.Value).First().Key).ToList();

            if (this.options.language == null)
            {
                tokens[.., this.sot_index + 1] = detectionResult.lang_tokens;
            }
        }

        return (languages, lang_probs);
    }

    private (Tensor tokens, Tensor sum_logprobs, List<float> no_speech_probs) MainLoop(Tensor audio_features, Tensor tokens)
    {
        int n_batch = (int)tokens.shape[0];
        Tensor sum_logprobs = torch.zeros(n_batch, device: audio_features.device);
        List<float> no_speech_probs = Enumerable.Repeat(float.NaN, n_batch).ToList();

        try
        {
            for (int i = 0; i < this.sample_len; i++)
            {
                Tensor logits = this.inference.logits(tokens, audio_features);
                
                if (i == 0 && this.tokenizer.no_speech.HasValue)
                {
                    Tensor probs_at_sot = logits[.., this.sot_index].@float().softmax(-1);
                    no_speech_probs = probs_at_sot[.., this.tokenizer.no_speech.Value].data<float>().ToList();
                }

                logits = logits[.., ^1];

                foreach (var logit_filter in this.logit_filters)
                {
                    logit_filter.apply(logits, tokens);
                }

                (tokens, bool completed) = this.decoder.update(tokens, logits, sum_logprobs);

                if (completed || tokens.shape[^1] > this.n_ctx)
                {
                    break;
                }
            }
        }
        finally
        {
            this.inference.cleanup_caching();
        }

        return (tokens, sum_logprobs, no_speech_probs);
    }

    public IEnumerable<DecodingResult> Run(Tensor mel)
    {
        using var _ = torch.no_grad();
        this.decoder.reset();
        int n_audio = (int)mel.shape[0];

        Tensor audio_features = GetAudioFeatures(mel);
        Tensor tokens = torch.tensor(this.initial_tokens).repeat(n_audio, 1);

        // detect language if requested, overwriting the language token.
        (List<string> languages, List<Dictionary<string, float>> language_probs) = DetectLanguage(audio_features, tokens);
        if (this.options.task == "lang_id")
        {
            foreach (var result in audio_features.AsEnumerable().Zip(languages, language_probs).Select(t => new DecodingResult(t.Item1, t.Item2, t.Item3)))
            {
                yield return result;
            }

            yield break;
        }

        // repeat text tensors by the group size, for beam search or best-of-n sampling
        tokens = tokens.repeat_interleave(this.n_group, dim: 0).to(audio_features.device);

        // call the main sampling loop
        (tokens, var sum_logprobs, var no_speech_probs) = this.MainLoop(audio_features, tokens);

        // reshape the tensors to have (n_audio, n_group) as the first two dimensions
        audio_features = audio_features.index_select(0, torch.arange(0, audio_features.shape[0], this.n_group, device: audio_features.device));
        no_speech_probs = no_speech_probs.Where((_, index) => index % this.n_group == 0).ToList();
        Debug.Assert(audio_features.shape[0] == no_speech_probs.Count() && audio_features.shape[0] == n_audio);
        tokens = tokens.reshape(n_audio, this.n_group, -1);
        sum_logprobs = sum_logprobs.reshape(n_audio, this.n_group);

        // get the final candidates for each group, and slice between the first sampled token and EOT
        var (tokens_decoded, sum_logprobs_decoded) = this.decoder.finalize(tokens, sum_logprobs);

        List<List<Tensor>> tokens1 = [
            ..from s in tokens_decoded.AsEnumerable()
                  select
                  (from t in s.AsEnumerable() select t[this.sample_begin..(int)t.eq(tokenizer.eot).nonzero()[0, 0]]).ToList()
        ];

        // select the top-ranked sample in each group
        var selected = this.sequence_ranker.Rank(tokens1, sum_logprobs_decoded);
        List<List<int>> tokens2 = [.. from it in selected.Zip(tokens1)
                                      select it.Item2[it.Item1].data<int>().ToList()];
        List<string> texts = [.. from t in tokens2 select tokenizer.Decode(t)?.Trim()];
        List<float> sum_logprobs2 = [.. from ilp in selected.Zip(sum_logprobs_decoded) select ilp.Item2[ilp.Item1]];
        List<float> avg_logprobs = [.. from tlp in tokens2.Zip(sum_logprobs2, (t, lp) => (t, lp)) select tlp.lp / (tlp.t.Count + 1)];


        var lengths = new[]
        {
            texts.Count,
            languages.Count,
            tokens.shape[0],
            audio_features.shape[0],
            avg_logprobs.Count,
            no_speech_probs.Count
        };
        Debug.Assert(lengths.Distinct().Count() == 1, $"inconsistent result lengths: {lengths.Stringify()}");

        for (int i = 0; i < texts.Count; i++)
        {
            yield return new DecodingResult(
                audio_features: audio_features[i],
                language: languages[i],
                tokens: tokens2[i],
                text: texts[i],
                avg_logprob: avg_logprobs[i],
                no_speech_prob: no_speech_probs[i],
                temperature: this.options.temperature,
                compression_ratio: compression_ratio(texts[i])
            );
        }
    }

    static float compression_ratio(string text)
    {
        var text_bytes = Encoding.UTF8.GetBytes(text);
        return text_bytes.Length / CompressWithZLib(text_bytes).Length;
    }

    static byte[] CompressWithZLib(byte[] data)
    {
        using (var decompressedStream = new MemoryStream(data))
        using (var compressedStream = new MemoryStream())
        using (var zlibStream = new ZLibStream(decompressedStream, CompressionMode.Compress))
        {
            zlibStream.CopyTo(compressedStream);
            return compressedStream.ToArray();
        }
    }
}
