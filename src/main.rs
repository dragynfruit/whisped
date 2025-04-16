use argh::FromArgs;
use reqwest;
use tokio::{fs, io::AsyncWriteExt};
use std::path::Path;
use hound;
use rubato::{FftFixedInOut, Resampler};
use minimp3::{Decoder, Frame, Error};
use lewton::inside_ogg::OggStreamReader;
use std::convert::TryInto;
use std::io::Cursor;
use whisper_rs::{WhisperContext, WhisperContextParameters, FullParams, SamplingStrategy};

/// Async command-line tool for Whisper ASR with auto model download
#[derive(FromArgs)]
struct Args {
    /// model to use (e.g., tiny, base, small, medium, large)
    #[argh(option, default = "String::from(\"base\")")]
    model: String,

    /// quantization level to use (e.g., q4_0, q5_0, q8_0). Default: no quantization.
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
    let ctx = WhisperContext::new_with_params(
        &model_path,
        WhisperContextParameters {
            flash_attn: true,
            ..Default::default()
        }
    ).expect("Failed to load model");
    
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
    
    // Run transcription
    let mut state = ctx.create_state().expect("Failed to create state");
    state.full(params, &audio_samples)
        .expect("Failed to run whisper model");
    
    println!("Transcription complete!");
    
    // Extract results
    let num_segments = state.full_n_segments()
        .expect("Failed to get number of segments");
    
    let mut full_text = String::new();
    
    // Extract results with or without timestamps based on the flag
    for i in 0..num_segments {
        let segment_text = state.full_get_segment_text(i)
            .expect("Failed to get segment text");
        
        if args.timestamps {
            let start_timestamp = state.full_get_segment_t0(i)
                .expect("Failed to get segment start timestamp");
            let end_timestamp = state.full_get_segment_t1(i)
                .expect("Failed to get segment end timestamp");
            
            // Format timestamps with explicit casting to ensure correct types
            let start_mins = (start_timestamp as f64 / 60.0) as i64;
            let start_secs = (start_timestamp as f64 % 60.0) as i64;
            let start_millis = ((start_timestamp as f64 * 1000.0) % 1000.0) as i64;
            
            let end_mins = (end_timestamp as f64 / 60.0) as i64;
            let end_secs = (end_timestamp as f64 % 60.0) as i64;
            let end_millis = ((end_timestamp as f64 * 1000.0) % 1000.0) as i64;
            
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

/// Downloads a model if it is not already cached and returns its local path.
async fn resolve_and_download_model(
    model_name: &str,
    quantization: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    const MODEL_BASE_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/";
    let model_url_map = vec!["tiny", "base", "small", "medium", "large"];

    // Validate model name
    if !model_url_map.contains(&model_name) {
        return Err(format!(
            "Invalid model name. Use one of: {}",
            model_url_map.join(", ")
        )
        .into());
    }

    // Construct model filename with optional quantization
    let model_filename = if quantization.is_empty() {
        format!("ggml-{}.bin", model_name)
    } else {
        format!("ggml-{}-{}.bin", model_name, quantization)
    };

    // Define cache directory
    let cache_dir = dirs::cache_dir()
        .ok_or("Could not determine cache directory")?
        .join("whisper_models");
    fs::create_dir_all(&cache_dir).await?; // Ensure the cache directory exists

    let model_path = cache_dir.join(&model_filename);

    // Check if the model is already cached
    if !model_path.exists() {
        println!("Downloading model '{}' (quant: '{}')...",
            model_name, 
            if quantization.is_empty() { "none" } else { quantization });
        
        let model_url = format!("{}{}", MODEL_BASE_URL, model_filename);

        // Download the model
        let response = reqwest::get(&model_url).await?;
        if response.status().is_success() {
            let total_size = response.content_length().unwrap_or(0);
            println!("Downloading {} bytes...", total_size);
            
            let bytes = response.bytes().await?;
            
            let mut file = fs::File::create(&model_path).await?;
            file.write_all(&bytes).await?;
            
            println!("Model '{}' downloaded successfully.", model_filename);
        } else {
            return Err(format!(
                "Failed to download model from '{}': {}",
                model_url,
                response.status()
            )
            .into());
        }
    } else {
        println!("Model '{}' already cached.", model_filename);
    }

    Ok(model_path.to_string_lossy().into_owned())
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
        let mut resampler =
            FftFixedInOut::<f32>::new(spec.sample_rate.try_into().unwrap(), 16_000, 1024, 2)?;
        let resampled = resampler
            .process(&[samples], None)?
            .into_iter()
            .flatten()
            .collect();
        return Ok(resampled);
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
        let mut resampler = FftFixedInOut::<f32>::new(
            sample_rate.try_into().unwrap(),
            16_000,
            1024,
            2,
        )?;
        let resampled = resampler
            .process(&[samples], None)?
            .into_iter()
            .flatten()
            .collect();
        return Ok(resampled);
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
        let mut resampler = FftFixedInOut::<f32>::new(
            reader.ident_hdr.audio_sample_rate.try_into().unwrap(),
            16_000,
            1024,
            2,
        )?;
        let resampled = resampler
            .process(&[samples], None)?
            .into_iter()
            .flatten()
            .collect();
        return Ok(resampled);
    }
    
    Ok(samples)
}