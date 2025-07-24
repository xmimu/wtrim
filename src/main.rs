use clap::Parser;
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use std::error::Error;

const FRAME_SIZE: usize = 1024;

#[derive(Parser, Debug)]
#[command(name = "wtrim", author, version, about = "WAV 裁剪及淡入淡出工具")]
struct Args {
    /// 输入 WAV 文件路径
    #[arg(short, long)]
    input: String,

    /// 输出 WAV 文件路径
    #[arg(short, long)]
    output: String,

    /// 裁剪阈值 (dB)，默认 -64 dB
    #[arg(short = 't', long, default_value_t = -64.0)]
    threshold_db: f32,

    /// 淡入淡出长度（样本数，单位采样点），默认 2205 (0.05秒 @ 44.1kHz)
    #[arg(short = 'f', long, default_value_t = 2205)]
    fade_samples: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let mut reader = WavReader::open(&args.input)?;
    let spec = reader.spec();
    let channels = spec.channels as usize;

    println!("读取文件：{}", args.input);
    println!(
        "格式：{} Hz, {} bit, {} 通道, {:?}",
        spec.sample_rate, spec.bits_per_sample, spec.channels, spec.sample_format
    );

    let samples = read_samples_to_f32(&mut reader, spec)?;

    // 将 dB 转换成线性振幅
    let threshold = db_to_amplitude(args.threshold_db);

    // 计算RMS帧（固定FRAME_SIZE）
    let rms_frames = samples_to_rms_frames(&samples, channels, FRAME_SIZE);

    // 找裁剪帧起止索引
    let (start_frame, end_frame) = trim_silence_rms_frames(&rms_frames, threshold);

    // 转成样本点索引区间
    let start_sample = start_frame * FRAME_SIZE * channels;
    let end_sample = ((end_frame + 1) * FRAME_SIZE * channels).min(samples.len());

    let trimmed_samples = &samples[start_sample..end_sample];

    // 转成帧方便淡入淡出（帧=采样点/声道数）
    let trimmed_frames = samples_to_frames(trimmed_samples, channels);

    // 淡入淡出处理
    let faded_frames = fade_in_out_frames(&trimmed_frames, args.fade_samples / channels);

    // 写回一维样本数组
    let output_samples = frames_to_samples(&faded_frames);

    write_wav(&args.output, &output_samples, spec)?;

    println!("处理完成，输出文件：{}", args.output);

    Ok(())
}

fn db_to_amplitude(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

fn read_samples_to_f32(
    reader: &mut WavReader<std::io::BufReader<std::fs::File>>,
    spec: WavSpec,
) -> Result<Vec<f32>, Box<dyn Error>> {
    match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, 16) => {
            let samples = reader
                .samples::<i16>()
                .filter_map(Result::ok)
                .map(|s| s as f32 / i16::MAX as f32)
                .collect();
            Ok(samples)
        }
        (SampleFormat::Int, 24 | 32) => {
            let bits = spec.bits_per_sample as u16;
            let samples = reader
                .samples::<i32>()
                .filter_map(Result::ok)
                .map(|s| s as f32 / ((1 << (bits - 1)) as f32))
                .collect();
            Ok(samples)
        }
        (SampleFormat::Float, 32) => {
            let samples = reader.samples::<f32>().filter_map(Result::ok).collect();
            Ok(samples)
        }
        _ => Err("Unsupported sample format".into()),
    }
}

fn rms_of_samples(samples: &[f32]) -> f32 {
    let sum_squares: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_squares / samples.len() as f32).sqrt()
}

fn samples_to_rms_frames(samples: &[f32], channels: usize, frame_size: usize) -> Vec<f32> {
    samples
        .chunks(frame_size * channels)
        .map(|chunk| rms_of_samples(chunk))
        .collect()
}

fn trim_silence_rms_frames(rms_frames: &[f32], threshold: f32) -> (usize, usize) {
    let start = rms_frames
        .iter()
        .position(|&rms| rms > threshold)
        .unwrap_or(0);

    let end = rms_frames
        .iter()
        .rposition(|&rms| rms > threshold)
        .unwrap_or(rms_frames.len() - 1);

    (start, end)
}

fn samples_to_frames(samples: &[f32], channels: usize) -> Vec<Vec<f32>> {
    samples
        .chunks(channels)
        .map(|frame| frame.to_vec())
        .collect()
}

fn fade_in_out_frames(frames: &[Vec<f32>], fade_samples: usize) -> Vec<Vec<f32>> {
    let len = frames.len();
    let mut out = Vec::with_capacity(len);

    for i in 0..len {
        let amp = if i < fade_samples {
            i as f32 / fade_samples as f32
        } else if i >= len - fade_samples {
            (len - i) as f32 / fade_samples as f32
        } else {
            1.0
        };

        let frame: Vec<f32> = frames[i].iter().map(|&s| s * amp).collect();
        out.push(frame);
    }

    out
}

fn frames_to_samples(frames: &[Vec<f32>]) -> Vec<f32> {
    frames.iter().flatten().copied().collect()
}

fn write_wav(path: &str, samples: &[f32], spec: WavSpec) -> Result<(), Box<dyn Error>> {
    let mut writer = WavWriter::create(path, spec)?;

    match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, 16) => {
            for &s in samples {
                let val = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                writer.write_sample(val)?;
            }
        }
        (SampleFormat::Int, 24 | 32) => {
            let bits = spec.bits_per_sample as u16;
            for &s in samples {
                let max_val = (1 << (bits - 1)) as f32;
                let val = (s.clamp(-1.0, 1.0) * max_val) as i32;
                writer.write_sample(val)?;
            }
        }
        (SampleFormat::Float, 32) => {
            for &s in samples {
                writer.write_sample(s)?;
            }
        }
        _ => return Err("Unsupported sample format".into()),
    }

    writer.finalize()?;
    Ok(())
}
