export type Json = string | number | boolean | null | { [key: string]: Json } | Json[]

export interface Database {
  public: {
    Tables: {
      profiles: {
        Row: { id: string; email: string; full_name: string | null; avatar_url: string | null; created_at: string }
        Insert: { id: string; email: string; full_name?: string | null; avatar_url?: string | null; created_at?: string }
        Update: { id?: string; email?: string; full_name?: string | null; avatar_url?: string | null; created_at?: string }
      }
      runs: {
        Row: {
          id: string; user_id: string; status: string; task_type: string | null
          model_id: string | null; dataset_filename: string | null; dataset_rows: number | null
          intent_spec: Json; model_recipe: Json; metrics: Json
          artifact_path: string | null; hf_model_url: string | null; error_message: string | null
          created_at: string; completed_at: string | null
          model_card: string | null; deploy_status: string | null; hf_repo_id: string | null
        }
        Insert: {
          id?: string; user_id: string; status?: string; task_type?: string | null
          model_id?: string | null; dataset_filename?: string | null; dataset_rows?: number | null
          intent_spec?: Json; model_recipe?: Json; metrics?: Json
          artifact_path?: string | null; hf_model_url?: string | null; error_message?: string | null
          created_at?: string; completed_at?: string | null
          model_card?: string | null; deploy_status?: string | null; hf_repo_id?: string | null
        }
        Update: {
          id?: string; user_id?: string; status?: string; task_type?: string | null
          model_id?: string | null; dataset_filename?: string | null; dataset_rows?: number | null
          intent_spec?: Json; model_recipe?: Json; metrics?: Json
          artifact_path?: string | null; hf_model_url?: string | null; error_message?: string | null
          created_at?: string; completed_at?: string | null
          model_card?: string | null; deploy_status?: string | null; hf_repo_id?: string | null
        }
      }
      run_events: {
        Row: { id: number; run_id: string; event_type: string; data: Json; created_at: string }
        Insert: { id?: number; run_id: string; event_type: string; data?: Json; created_at?: string }
        Update: { id?: number; run_id?: string; event_type?: string; data?: Json; created_at?: string }
      }
    }
  }
}

export type Run = Database["public"]["Tables"]["runs"]["Row"]
export type RunEvent = Database["public"]["Tables"]["run_events"]["Row"]
export type Profile = Database["public"]["Tables"]["profiles"]["Row"]
