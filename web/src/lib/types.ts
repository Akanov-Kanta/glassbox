import type { Model } from '../engine/types';

/** Shape of web/public/model/model.json (see SPEC.md). */
export interface ModelMeta {
  vocab: string[];
  n_layer: number;
  n_head: number;
  n_embd: number;
  block_size: number;
  dtype: string;
  params: { name: string; shape: number[]; offset: number }[];
}

/** Shape of web/public/model/training.json (see SPEC.md). Extra fields are optional. */
export interface TrainingReport {
  corpus: { name: string; source_url?: string; license?: string; chars?: number };
  params: number;
  device: string;
  train_seconds: number;
  steps: number[];
  train_loss: number[];
  val_loss: number[];
  final: { train_loss: number; val_loss: number };
  samples: string[];
  optimizer?: string;
  batch_size?: number;
  learning_rate?: number;
  lr?: number;
}

export type DataSource = 'real' | 'mock';

export interface LoadedApp {
  model: Model;
  training: TrainingReport;
  /** where the model + weights came from */
  modelSource: DataSource;
  /** where the training report came from */
  trainingSource: DataSource;
  /** size of the weight file in bytes (real or estimated) */
  weightBytes: number;
}

export interface LoadProgress {
  phase: 'meta' | 'weights' | 'init';
  loaded: number;
  total: number;
}
