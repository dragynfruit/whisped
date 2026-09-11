use argh::FromArgs;
use reqwest;
use tokio::{fs, io::{AsyncReadExt, AsyncWriteExt}};
use std::path::Path;
use hound;
use rubato::{Fft, FixedSync, Resampler, audioadapter_buffers::owned::InterleavedOwned};
use minimp3::{Decoder, Frame, Error};
use lewton::inside_ogg::OggStreamReader;
use std::io::Cursor;
use whisper_rs::{WhisperContext, WhisperContextParameters, FullParams, SamplingStrategy};

/// Async command-line tool for Whisper ASR with auto model download
#[derive(FromArgs)]
struct Args {
    /// model to use (e.g., tiny, base, small.en, medium, large-v3, large-v3-turbo)
    #[argh(option, default = "String::from(\"base\")")]
    model: String,

    /// quantization level to use (q5_0, q5_1, or q8_0). Default: no quantization.
    #[argh(option, default = "String::new()")]
    quant: String,

    /// path to the input audio file
    #[argh(option)]
    input: String,

    /// path to the output text file
    #[argh(option)]
    output: String,

    /// language code (optional, e.g., en, fr, de). Default: auto-detect
    #[argh(option, default = "String::new()")]
    language: String,

    /// enable text translation to English
    #[argh(switch)]
    translate: bool,
    
    /// include timestamps in the output
    #[argh(switch)]
    timestamps: bool,

    /// use GPU acceleration (requires a build with the cuda or vulkan feature)
    #[argh(switch)]
    gpu: bool,

    /// GPU device index to use
    #[argh(option, default = "0")]
    gpu_device: i32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    // Resolve and download model if needed
    println!("Resolving model...");
    let model_path = resolve_and_download_model(&args.model, &args.quant).await?;
    println!("Model ready at: {}", model_path);

    // Parse the audio file into a format suitable for Whisper
    println!("Processing audio file '{}'...", args.input);
    let audio_samples = parse_audio_data(&args.input).await?;
    println!("Audio processed: {} samples", audio_samples.len());

    // Transcribe the audio
    println!("Transcribing using model '{}' (quant: '{}')...",
        args.model, 
        if args.quant.is_empty() { "none" } else { &args.quant });

    // Load the whisper context with the model
    let mut ctx_params = WhisperContextParameters {
        flash_attn: true,
        ..Default::default()
    };

    if args.gpu {
        #[cfg(not(any(feature = "cuda", feature = "vulkan")))]
        println!("Warning: --gpu ignored, this build has no GPU backend. Rebuild with --features cuda or --features vulkan.");
        ctx_params.use_gpu(true).gpu_device(args.gpu_device);
    }

    let ctx = WhisperContext::new_with_params(&model_path, ctx_params)
        .expect("Failed to load model");
    
    // Create processing parameters
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    
    // Set language if specified
    if !args.language.is_empty() {
        params.set_language(Some(&args.language));
    }
    
    // Set translation flag if enabled
    if args.translate {
        params.set_translate(true);
    }

    // Enable talk diarization for tdrz models
    if args.model.ends_with("-tdrz") {
        params.set_tdrz_enable(true);
    }
    
    // Run transcription
    let mut state = ctx.create_state().expect("Failed to create state");
    state.full(params, &audio_samples)
        .expect("Failed to run whisper model");
    
    println!("Transcription complete!");
    
    // Extract results
    let num_segments = state.full_n_segments();    
    let mut full_text = String::new();
    
    // Extract results with or without timestamps based on the flag
    for i in 0..num_segments {
        let segment = state
            .get_segment(i)
            .expect("Failed to get segment");
        let segment_text = segment.to_str()?;
        
        if args.timestamps {
            // Timestamps are in centiseconds (10s of milliseconds)
            let start_ms = segment.start_timestamp() * 10;
            let end_ms = segment.end_timestamp() * 10;

            let start_mins = start_ms / 60_000;
            let start_secs = (start_ms / 1000) % 60;
            let start_millis = start_ms % 1000;

            let end_mins = end_ms / 60_000;
            let end_secs = (end_ms / 1000) % 60;
            let end_millis = end_ms % 1000;
            
            full_text.push_str(&format!(
                "[{:02}:{:02}.{:03} - {:02}:{:02}.{:03}] {}\n",
                start_mins, start_secs, start_millis,
                end_mins, end_secs, end_millis,
                segment_text
            ));
        } else {
            full_text.push_str(&segment_text);
            full_text.push_str(" ");
        }
    }
    
    // Save text to output file
    let output_file = &args.output;
    fs::write(output_file, if args.timestamps { &full_text } else { full_text.trim() }).await?;
    
    println!("Transcription saved to '{}'", output_file);
    if args.timestamps {
        println!("Output includes timestamps as requested");
    }

    Ok(())
}

/// Available ggml models and their SHA-1 checksums.
const MODELS: &[(&str, &str)] = &[
    ("tiny", "bd577a113a864445d4c299885e0cb97d4ba92b5f"),
    ("tiny-q5_1", "2827a03e495b1ed3048ef28a6a4620537db4ee51"),
    ("tiny-q8_0", "19e8118f6652a650569f5a949d962154e01571d9"),
    ("tiny.en", "c78c86eb1a8faa21b369bcd33207cc90d64ae9df"),
    ("tiny.en-q5_1", "3fb92ec865cbbc769f08137f22470d6b66e071b6"),
    ("tiny.en-q8_0", "802d6668e7d411123e672abe4cb6c18f12306abb"),
    ("base", "465707469ff3a37a2b9b8d8f89f2f99de7299dac"),
    ("base-q5_1", "a3733eda680ef76256db5fc5dd9de8629e62c5e7"),
    ("base-q8_0", "7bb89bb49ed6955013b9c20ae49423c94a20fbe"),
    ("base.en", "137c40403d78fd54d454da0f9bd998f78703390c"),
    ("base.en-q5_1", "d26d7ce5a1b6e57bea5d0431b9c20ae49423c94a"),
    ("base.en-q8_0", "bb1574182e9b924452bf0cd1510ac034d323e948"),
    ("small", "55356645c2b361a969dfd0ef2c5a50d530afd8d5"),
    ("small-q5_1", "6fe57ddcfdd1c6b07cdcc73aaf620810ce5fc771"),
    ("small-q8_0", "bcad8a2083f4e53d648d586b7dbc0cd673d8afad"),
    ("small.en", "db8a495a91d927739e50b3fc1cc4c6b8f6c2d022"),
    ("small.en-q5_1", "20f54878d608f94e4a8ee3ae56016571d47cba34"),
    ("small.en-q8_0", "9d75ff4ccfa0a8217870d7405cf8cef0a5579852"),
    ("small.en-tdrz", "b6c6e7e89af1a35c08e6de56b66ca6a02a2fdfa1"),
    ("medium", "fd9727b6e1217c2f614f9b698455c4ffd82463b4"),
    ("medium-q5_0", "7718d4c1ec62ca96998f058114db98236937490e"),
    ("medium-q8_0", "e66645948aff4bebbec71b3485c576f3d63af5d6"),
    ("medium.en", "8c30f0e44ce9560643ebd10bbe50cd20eafd3723"),
    ("medium.en-q5_0", "bb3b5281bddd61605d6fc76bc5b92d8f20284c3b"),
    ("medium.en-q8_0", "b1cf48c12c807e14881f634fb7b6c6ca867f6b38"),
    ("large-v1", "b1caaf735c4cc1429223d5a74f0f4d0b9b59a299"),
    ("large-v2", "0f4c8e34f21cf1a914c59d8b3ce882345ad349d6"),
    ("large-v2-q5_0", "00e39f2196344e901b3a2bd5814807a769bd1630"),
    ("large-v2-q8_0", "da97d6ca8f8ffbeeb5fd147f79010eeea194ba38"),
    ("large-v3", "ad82bf6a9043ceed055076d0fd39f5f186ff8062"),
    ("large-v3-q5_0", "e6e2ed78495d403bef4b7cff42ef4aaadcfea8de"),
    ("large-v3-turbo", "4af2b29d7ec73d781377bfd1758ca957a807e941"),
    ("large-v3-turbo-q5_0", "e050f7970618a659205450ad97eb95a18d69c9ee"),
    ("large-v3-turbo-q8_0", "01bf15bedffe9f39d65c1b6ff9b687ea91f59e0e"),
];

fn find_model_sha(name: &str) -> Option<&'static str> {
    MODELS
        .iter()
        .find(|(model, _)| *model == name)
        .map(|(_, sha)| *sha)
}

/// Downloads a model if it is not already cached and returns its local path.
async fn resolve_and_download_model(
    model_name: &str,
    quantization: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    const MODEL_BASE_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/";

    let full_name = if quantization.is_empty() {
        model_name.to_string()
    } else {
        format!("{}-{}", model_name, quantization)
    };

    // Validate model against the known lineup
    let expected_sha = find_model_sha(&full_name).ok_or_else(|| {
        format!(
            "Invalid model '{}'. Available models: {}",
            full_name,
            MODELS
                .iter()
                .map(|(name, _)| *name)
                .collect::<Vec<_>>()
                .join(", ")
        )
    })?;

    let model_filename = format!("ggml-{}.bin", full_name);

    // Define cache directory
    let cache_dir = dirs::cache_dir()
        .ok_or("Could not determine cache directory")?
        .join("whisper_models");
    fs::create_dir_all(&cache_dir).await?; // Ensure the cache directory exists

    let model_path = cache_dir.join(&model_filename);

    // Check if the model is already cached and valid
    if model_path.exists() && verify_sha1(&model_path, expected_sha).await? {
        println!("Model '{}' already cached.", model_filename);
        return Ok(model_path.to_string_lossy().into_owned());
    }

    if model_path.exists() {
        println!(
            "Cached model '{}' failed checksum verification, re-downloading.",
            model_filename
        );
    } else {
        println!("Downloading model '{}'...", model_filename);
    }

    let model_url = format!("{}{}", MODEL_BASE_URL, model_filename);

    // Download the model
    let response = reqwest::get(&model_url).await?;
    if !response.status().is_success() {
        return Err(format!(
            "Failed to download model from '{}': {}",
            model_url,
            response.status()
        )
        .into());
    }

    let total_size = response.content_length().unwrap_or(0);
    println!("Downloading {} bytes...", total_size);

    let bytes = response.bytes().await?;

    // Verify the checksum before writing to the cache
    let actual_sha = sha1_hex(&bytes);
    if actual_sha != expected_sha {
        return Err(format!(
            "Checksum mismatch for '{}': expected {}, got {}",
            model_url, expected_sha, actual_sha
        )
        .into());
    }

    let mut file = fs::File::create(&model_path).await?;
    file.write_all(&bytes).await?;

    println!("Model '{}' downloaded successfully.", model_filename);

    Ok(model_path.to_string_lossy().into_owned())
}

fn sha1_hex(bytes: &[u8]) -> String {
    use sha1::{Digest, Sha1};
    let mut hasher = Sha1::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

async fn verify_sha1(path: &Path, expected_sha: &str) -> Result<bool, Box<dyn std::error::Error>> {
    use sha1::{Digest, Sha1};
    let mut hasher = Sha1::new();
    let mut file = fs::File::open(path).await?;
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer).await?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()) == expected_sha)
}

/// Parses an audio file into a vector of floating-point samples (16 kHz, mono).
async fn parse_audio_data(file_path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let extension = Path::new(file_path)
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        .unwrap_or("")
        .to_lowercase();

    match extension.as_str() {
        "wav" => parse_wav_data(file_path).await,
        "mp3" => parse_mp3_data(file_path).await,
        "ogg" => parse_ogg_data(file_path).await,
        _ => Err(format!("Unsupported audio format: {}", extension).into()),
    }
}

/// Parses a WAV file into a vector of floating-point samples (16 kHz, mono).
async fn parse_wav_data(file_path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(file_path)?;
    let spec = reader.spec();
    let channels = spec.channels as usize;

    if spec.sample_format != hound::SampleFormat::Int {
        return Err("Only PCM WAV files are supported.".into());
    }

    // Read samples and convert to mono if needed
    let mut samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect();

    // Convert stereo to mono if needed
    if channels == 2 {
        let mut mono_samples = Vec::with_capacity(samples.len() / 2);
        for i in (0..samples.len()).step_by(2) {
            if i + 1 < samples.len() {
                mono_samples.push((samples[i] + samples[i + 1]) / 2.0);
            } else {
                mono_samples.push(samples[i]);
            }
        }
        samples = mono_samples;
    }

    if spec.sample_rate != 16_000 {
        return Ok(resample_to_16k(samples, spec.sample_rate)?);
    }

    Ok(samples)
}

/// Parses an MP3 file into a vector of floating-point samples (16 kHz, mono).
async fn parse_mp3_data(file_path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // Read the file contents asynchronously
    let file_data = fs::read(file_path).await?;

    // We need to process MP3 synchronously since minimp3 doesn't have async APIs
    let mut decoder = Decoder::new(std::io::Cursor::new(file_data));

    let mut samples = Vec::new();
    let mut sample_rate = 0;
    
    loop {
        match decoder.next_frame() {
            Ok(Frame {
                data,
                sample_rate: sr,
                channels: ch,
                ..
            }) => {
                sample_rate = sr;
                let ch_count = ch as usize;
                
                // Convert to mono if stereo
                if ch_count == 2 {
                    for i in (0..data.len()).step_by(2) {
                        if i + 1 < data.len() {
                            samples.push(((data[i] + data[i + 1]) / 2) as f32 / i16::MAX as f32);
                        } else {
                            samples.push(data[i] as f32 / i16::MAX as f32);
                        }
                    }
                } else {
                    samples.extend(data.into_iter().map(|s| s as f32 / i16::MAX as f32));
                }
            }
            Err(Error::Eof) => break,
            Err(e) => return Err(Box::new(e)),
        }
    }

    if sample_rate != 16_000 {
        return Ok(resample_to_16k(samples, sample_rate as u32)?);
    }

    Ok(samples)
}

/// Parses an OGG file into a vector of floating-point samples (16 kHz, mono).
/// Uses async for file I/O but synchronous processing for the actual decoding.
async fn parse_ogg_data(file_path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // Read the entire file asynchronously
    let file_data = fs::read(file_path).await?;
    
    // Process the OGG data synchronously
    let cursor = Cursor::new(file_data);
    let mut reader = OggStreamReader::new(cursor)?;
    
    let channels = reader.ident_hdr.audio_channels as usize;
    let mut samples = Vec::new();
    
    // Read and decode packets synchronously
    while let Some(packet) = reader.read_dec_packet_itl()? {
        // Convert to mono if stereo
        if channels == 2 {
            for i in (0..packet.len()).step_by(2) {
                if i + 1 < packet.len() {
                    samples.push((packet[i] as f32 + packet[i + 1] as f32) / (2.0 * i16::MAX as f32));
                } else {
                    samples.push(packet[i] as f32 / i16::MAX as f32);
                }
            }
        } else {
            samples.extend(packet.into_iter().map(|s| s as f32 / i16::MAX as f32));
        }
    }
    
    // Resample if necessary
    if reader.ident_hdr.audio_sample_rate != 16_000 {
        return Ok(resample_to_16k(samples, reader.ident_hdr.audio_sample_rate as u32)?);
    }
    
    Ok(samples)
}

/// Resamples mono f32 samples from the given sample rate to 16 kHz.
fn resample_to_16k(samples: Vec<f32>, sample_rate: u32) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let frames = samples.len();
    let input = InterleavedOwned::new_from(samples, 1, frames)?;
    let mut resampler =
        Fft::<f32>::new(sample_rate as usize, 16_000, 1024, 1, FixedSync::Input)?;
    let output = resampler.process_all(&input, frames, None)?;
    Ok(output.take_data())
}