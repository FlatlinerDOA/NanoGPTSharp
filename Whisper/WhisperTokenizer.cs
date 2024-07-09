using TorchSharp.Modules;
namespace PerceptivePyro.Whisper;

public class WhisperTokenizer
{
    public static IReadOnlyDictionary<string, string> Languages = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
    {
        ["en"] = "english",
        ["zh"] = "chinese",
        ["de"] = "german",
        ["es"] = "spanish",
        ["ru"] = "russian",
        ["ko"] = "korean",
        ["fr"] = "french",
        ["ja"] = "japanese",
        ["pt"] = "portuguese",
        ["tr"] = "turkish",
        ["pl"] = "polish",
        ["ca"] = "catalan",
        ["nl"] = "dutch",
        ["ar"] = "arabic",
        ["sv"] = "swedish",
        ["it"] = "italian",
        ["id"] = "indonesian",
        ["hi"] = "hindi",
        ["fi"] = "finnish",
        ["vi"] = "vietnamese",
        ["he"] = "hebrew",
        ["uk"] = "ukrainian",
        ["el"] = "greek",
        ["ms"] = "malay",
        ["cs"] = "czech",
        ["ro"] = "romanian",
        ["da"] = "danish",
        ["hu"] = "hungarian",
        ["ta"] = "tamil",
        ["no"] = "norwegian",
        ["th"] = "thai",
        ["ur"] = "urdu",
        ["hr"] = "croatian",
        ["bg"] = "bulgarian",
        ["lt"] = "lithuanian",
        ["la"] = "latin",
        ["mi"] = "maori",
        ["ml"] = "malayalam",
        ["cy"] = "welsh",
        ["sk"] = "slovak",
        ["te"] = "telugu",
        ["fa"] = "persian",
        ["lv"] = "latvian",
        ["bn"] = "bengali",
        ["sr"] = "serbian",
        ["az"] = "azerbaijani",
        ["sl"] = "slovenian",
        ["kn"] = "kannada",
        ["et"] = "estonian",
        ["mk"] = "macedonian",
        ["br"] = "breton",
        ["eu"] = "basque",
        ["is"] = "icelandic",
        ["hy"] = "armenian",
        ["ne"] = "nepali",
        ["mn"] = "mongolian",
        ["bs"] = "bosnian",
        ["kk"] = "kazakh",
        ["sq"] = "albanian",
        ["sw"] = "swahili",
        ["gl"] = "galician",
        ["mr"] = "marathi",
        ["pa"] = "punjabi",
        ["si"] = "sinhala",
        ["km"] = "khmer",
        ["sn"] = "shona",
        ["yo"] = "yoruba",
        ["so"] = "somali",
        ["af"] = "afrikaans",
        ["oc"] = "occitan",
        ["ka"] = "georgian",
        ["be"] = "belarusian",
        ["tg"] = "tajik",
        ["sd"] = "sindhi",
        ["gu"] = "gujarati",
        ["am"] = "amharic",
        ["yi"] = "yiddish",
        ["lo"] = "lao",
        ["uz"] = "uzbek",
        ["fo"] = "faroese",
        ["ht"] = "haitian creole",
        ["ps"] = "pashto",
        ["tk"] = "turkmen",
        ["nn"] = "nynorsk",
        ["mt"] = "maltese",
        ["sa"] = "sanskrit",
        ["lb"] = "luxembourgish",
        ["my"] = "myanmar",
        ["bo"] = "tibetan",
        ["tl"] = "tagalog",
        ["mg"] = "malagasy",
        ["as"] = "assamese",
        ["tt"] = "tatar",
        ["haw"] = "hawaiian",
        ["ln"] = "lingala",
        ["ha"] = "hausa",
        ["ba"] = "bashkir",
        ["jw"] = "javanese",
        ["su"] = "sundanese",
        ["yue"] = "cantonese",
    };

    public static readonly IReadOnlyDictionary<string, string> ToLanguageCode = new Dictionary<string, string>(
            Languages.Select(kv => new KeyValuePair<string, string>(kv.Value, kv.Key)).Concat(
            [
                new("burmese", "my"),
                new("valencian", "ca"),
                new("flemish", "nl"),
                new("haitian", "ht"),
                new("letzeburgesch", "lb"),
                new("pushto", "ps"),
                new("panjabi", "pa"),
                new("moldavian", "ro"),
                new("moldovan", "ro"),
                new("sinhalese", "si"),
                new("castilian", "es"),
                new("mandarin", "zh"),
            ]));

    private readonly WhisperEncoding encoding;
    private readonly int num_languages;
    private readonly string? language;
    private readonly string? task;
    private readonly Dictionary<string, int> special_tokens;

    public WhisperTokenizer(WhisperEncoding encoding, int num_languages, string? language = null, string? task = null, int[]? sot_sequence = null, Dictionary<string, int>? special_tokens = null)
    {
        this.encoding = encoding;
        this.num_languages = num_languages;
        this.language = language;
        this.task = task;
        this.sot_sequence = sot_sequence ?? Array.Empty<int>();

        // IMPORTANT: YOU need to load the GptEncoding's specail tokens into special_tokens yourself as SharpToken doesn't expose it's dictionary.
        //for special in self.encoding.special_tokens_set:
        //    special_token = self.encoding.encode_single_token(special)
        //    self.special_tokens[special] = special_token

        this.special_tokens = special_tokens ?? new Dictionary<string, int>();

        this.transcribe = this.special_tokens["<|transcribe|>"];
        this.translate = this.special_tokens["<|translate|>"];
        this.sot = this.special_tokens["<|startoftranscript|>"];
        this.sot_lm = this.special_tokens["<|startoflm|>"];
        this.sot_prev = this.special_tokens["<|startofprev|>"];
        this.no_speech = this.special_tokens["<|notimestamps|>"];
        this.no_timestamps = this.special_tokens["<|notimestamps|>"];
        this.timestamp_begin = this.special_tokens["<|0.00|>"];

        var langs = Languages.Keys.Take(this.num_languages).ToList();

        sot_sequence = [this.sot];
        if (this.language is not null)
        {
            sot_sequence = sot_sequence.Append(this.sot + 1 + langs.IndexOf(this.language)).ToArray();
            if (this.task is not null)
            {
                var task_token = this.task == "transcribe" ? this.transcribe : this.translate;
                sot_sequence.Append(task_token);

                this.sot_sequence = [.. sot_sequence];
            }
        }

        this.language_token = this.language is not null ? this.to_language_token(this.language) : throw new InvalidOperationException("This tokenizer does not have language token configured");
        this.all_language_tokens = (from tokenAndId in this.special_tokens
                                    let token = tokenAndId.Key
                                    let id = tokenAndId.Value
                                    where Languages.ContainsKey(token.Trim('<', '|', '>'))
                                    select id).ToArray();
        this.all_language_codes = (from id in this.all_language_tokens
                                  select this.Decode([id])!.Trim('<', '|', '>')).ToArray();
        this.sot_sequence_including_notimestamps = this.sot_sequence.Append(this.no_timestamps).ToArray();
    }


    public static WhisperTokenizer GetTokenizer(bool multilingual, int num_languages, string? language = null, string? task = null)
    {
        // TODO: Is lru_cache needed?
        if (language is not null)
        {
            language = language.ToLower();
            if (!Languages.ContainsKey(language))
            {
                if (ToLanguageCode.ContainsKey(language))
                {
                    language = ToLanguageCode[language];
                }
                else
                {
                    throw new ArgumentException($"Unsupported language: {language}");
                }
            }
        }

        string encoding_name;
        if (multilingual)
        {
            encoding_name = "multilingual";
            language ??= "en";
            task ??= "transcribe";
        }
        else
        {
            encoding_name = "gpt2";
            language = null;
            task = null;
        }


        var encoding = WhisperEncoding.GetEncoding(encoding_name, num_languages);
        return new WhisperTokenizer(encoding, num_languages, language, task);
    }

    public int[] sot_sequence { get; }

    public int eot => this.encoding.eot_token;

    public int transcribe { get; init; }

    public int translate { get; init; }

    public int sot { get; init; }

    public int sot_lm { get; init; }

    public int sot_prev { get; init; }

    public int? no_speech { get; init; }

    public int no_timestamps { get; init; }

    public int timestamp_begin { get; init; }

    public int language_token { get; init; }

    public int[] all_language_tokens { get; init; }

    public string[] all_language_codes { get; init; }

    public int[] sot_sequence_including_notimestamps { get; init; }

    public int[] non_speech_tokens { get; init; }

    public int to_language_token(string language) => this.special_tokens[$"<|{language}|>"];


    public IReadOnlyList<int> Encode(string text) => this.encoding.Encode(text);

    public string? Decode(List<int> token_ids)
    {
        token_ids = [.. from t in token_ids where t < this.timestamp_begin select t];
        return this.encoding.Decode(token_ids);
    }

    /// <summary>
    /// Timestamp tokens are above other special tokens' id range and are ignored by `decode()`.
    /// This method decodes given tokens with timestamps tokens annotated, e.g. "<|1.08|>".
    /// </summary>
    /// <param name=""></param>
    /// <param name=""></param>
    /// <param name=""></param>
    /// <param name="kwargs"></param>
    /// <returns></returns>
    string? decode_with_timestamps(List<int> token_ids)
    {
        return this.encoding.Decode(token_ids);
    }
}
