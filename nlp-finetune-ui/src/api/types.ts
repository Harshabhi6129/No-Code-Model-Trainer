export interface SuggestResp {
    task: string;
    recommended_model: string;
    reason: string;
  }
  export interface ValidateResp {
    valid: boolean;
    warnings: string[];
    suggestions: string[];
  }
  export interface HParamResp {
    epochs: number;
    batch_size: number;
    learning_rate: number;
    use_peft: boolean;
  }
  export interface TrainResp {
    run_id: string;
    status: "started";
  }
  