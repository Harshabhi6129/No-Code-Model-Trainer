// src/components/TrainingControlPanel.tsx
import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Typography,
  Slider,
  TextField,
  Grid,
  Chip,
  Alert,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  PlayArrow,
  Pause,
  Stop,
  Settings,
  CheckCircle,
  Error,
  Warning
} from '@mui/icons-material';
import axios from 'axios';

interface TrainingControlPanelProps {
  runId: string;
  onStatusChange?: (status: string) => void;
}

interface TrainingStatus {
  status: 'running' | 'paused' | 'stopped' | 'completed' | 'failed';
  epoch: number;
  batch: number;
  total_batches: number;
  elapsed_time: number;
}

interface ParamUpdate {
  learning_rate?: number;
  weight_decay?: number;
  dropout?: number;
  gradient_clip?: number;
}

const TrainingControlPanel: React.FC<TrainingControlPanelProps> = ({
  runId,
  onStatusChange
}) => {
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showParams, setShowParams] = useState(false);
  
  // Parameter states
  const [learningRate, setLearningRate] = useState(5e-5);
  const [weightDecay, setWeightDecay] = useState(0.01);
  const [dropout, setDropout] = useState(0.1);
  const [gradientClip, setGradientClip] = useState(1.0);

  // Fetch training status
  const fetchStatus = async () => {
    try {
      const response = await axios.get(`/api/training/${runId}/status`);
      setStatus(response.data);
      onStatusChange?.(response.data.status);
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to fetch status');
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 2000); // Poll every 2 seconds
    return () => clearInterval(interval);
  }, [runId]);

  const handlePause = async () => {
    setLoading(true);
    try {
      await axios.post(`/api/training/${runId}/pause`);
      await fetchStatus();
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to pause training');
    }
    setLoading(false);
  };

  const handleResume = async () => {
    setLoading(true);
    try {
      await axios.post(`/api/training/${runId}/resume`);
      await fetchStatus();
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to resume training');
    }
    setLoading(false);
  };

  const handleStop = async () => {
    if (!window.confirm('Are you sure you want to stop training? This cannot be undone.')) {
      return;
    }
    
    setLoading(true);
    try {
      await axios.post(`/api/training/${runId}/stop`);
      await fetchStatus();
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to stop training');
    }
    setLoading(false);
  };

  const handleUpdateParams = async () => {
    setLoading(true);
    try {
      const params: ParamUpdate = {
        learning_rate: learningRate,
        weight_decay: weightDecay,
        dropout: dropout,
        gradient_clip: gradientClip
      };
      
      await axios.post(`/api/training/${runId}/update-params`, params);
      setError(null);
      alert('Parameters updated successfully!');
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to update parameters');
    }
    setLoading(false);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'success';
      case 'paused': return 'warning';
      case 'stopped': return 'error';
      case 'completed': return 'info';
      case 'failed': return 'error';
      default: return 'default';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running': return <PlayArrow />;
      case 'paused': return <Pause />;
      case 'stopped': return <Stop />;
      case 'completed': return <CheckCircle />;
      case 'failed': return <Error />;
      default: return <Warning />;
    }
  };

  const formatTime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  if (!status) {
    return <Typography>Loading training status...</Typography>;
  }

  return (
    <Card>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6">Training Control</Typography>
          <Chip
            icon={getStatusIcon(status.status)}
            label={status.status.toUpperCase()}
            color={getStatusColor(status.status) as any}
            variant="outlined"
          />
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        <Grid container spacing={2} alignItems="center">
          <Grid item>
            <Typography variant="body2" color="textSecondary">
              Epoch: {status.epoch} | Batch: {status.batch}/{status.total_batches}
            </Typography>
          </Grid>
          <Grid item>
            <Typography variant="body2" color="textSecondary">
              Elapsed: {formatTime(status.elapsed_time)}
            </Typography>
          </Grid>
        </Grid>

        <Box display="flex" gap={1} mt={2} mb={2}>
          {status.status === 'running' && (
            <Button
              variant="contained"
              color="warning"
              startIcon={<Pause />}
              onClick={handlePause}
              disabled={loading}
            >
              Pause
            </Button>
          )}
          
          {status.status === 'paused' && (
            <Button
              variant="contained"
              color="success"
              startIcon={<PlayArrow />}
              onClick={handleResume}
              disabled={loading}
            >
              Resume
            </Button>
          )}
          
          {(status.status === 'running' || status.status === 'paused') && (
            <Button
              variant="contained"
              color="error"
              startIcon={<Stop />}
              onClick={handleStop}
              disabled={loading}
            >
              Stop
            </Button>
          )}

          <Tooltip title="Adjust Parameters">
            <IconButton
              onClick={() => setShowParams(!showParams)}
              disabled={status.status !== 'running'}
            >
              <Settings />
            </IconButton>
          </Tooltip>
        </Box>

        {showParams && status.status === 'running' && (
          <Card variant="outlined" sx={{ mt: 2 }}>
            <CardContent>
              <Typography variant="subtitle2" gutterBottom>
                Adjust Parameters (Live)
              </Typography>
              
              <Grid container spacing={3}>
                <Grid item xs={6}>
                  <Typography gutterBottom>Learning Rate</Typography>
                  <Slider
                    value={learningRate}
                    onChange={(_, value) => setLearningRate(value as number)}
                    min={1e-6}
                    max={1e-3}
                    step={1e-6}
                    scale={(x) => x}
                    valueLabelDisplay="auto"
                    valueLabelFormat={(value) => value.toExponential(2)}
                  />
                  <TextField
                    size="small"
                    type="number"
                    value={learningRate}
                    onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                    inputProps={{ step: 1e-6, min: 1e-6, max: 1e-3 }}
                  />
                </Grid>

                <Grid item xs={6}>
                  <Typography gutterBottom>Weight Decay</Typography>
                  <Slider
                    value={weightDecay}
                    onChange={(_, value) => setWeightDecay(value as number)}
                    min={0}
                    max={0.1}
                    step={0.001}
                    valueLabelDisplay="auto"
                  />
                  <TextField
                    size="small"
                    type="number"
                    value={weightDecay}
                    onChange={(e) => setWeightDecay(parseFloat(e.target.value))}
                    inputProps={{ step: 0.001, min: 0, max: 0.1 }}
                  />
                </Grid>

                <Grid item xs={6}>
                  <Typography gutterBottom>Dropout</Typography>
                  <Slider
                    value={dropout}
                    onChange={(_, value) => setDropout(value as number)}
                    min={0}
                    max={0.5}
                    step={0.01}
                    valueLabelDisplay="auto"
                  />
                  <TextField
                    size="small"
                    type="number"
                    value={dropout}
                    onChange={(e) => setDropout(parseFloat(e.target.value))}
                    inputProps={{ step: 0.01, min: 0, max: 0.5 }}
                  />
                </Grid>

                <Grid item xs={6}>
                  <Typography gutterBottom>Gradient Clip</Typography>
                  <Slider
                    value={gradientClip}
                    onChange={(_, value) => setGradientClip(value as number)}
                    min={0.1}
                    max={5.0}
                    step={0.1}
                    valueLabelDisplay="auto"
                  />
                  <TextField
                    size="small"
                    type="number"
                    value={gradientClip}
                    onChange={(e) => setGradientClip(parseFloat(e.target.value))}
                    inputProps={{ step: 0.1, min: 0.1, max: 5.0 }}
                  />
                </Grid>
              </Grid>

              <Button
                variant="contained"
                onClick={handleUpdateParams}
                disabled={loading}
                sx={{ mt: 2 }}
              >
                Apply Changes
              </Button>
            </CardContent>
          </Card>
        )}
      </CardContent>
    </Card>
  );
};

export default TrainingControlPanel;