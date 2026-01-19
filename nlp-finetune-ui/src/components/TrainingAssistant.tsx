// src/components/TrainingAssistant.tsx
import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Alert,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  CircularProgress,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  Psychology,
  TrendingUp,
  Warning,
  CheckCircle,
  Error,
  AutoFixHigh,
  Refresh,
  Lightbulb
} from '@mui/icons-material';
import axios from 'axios';

interface TrainingAnalysis {
  status: 'healthy' | 'overfitting' | 'underfitting' | 'diverging' | 'plateau' | 'insufficient_data' | 'analysis_failed';
  confidence?: number;
  issues?: string[];
  suggestions?: string[];
  auto_actions?: Array<{
    action: string;
    value?: number;
    reason: string;
  }>;
  error?: string;
}

interface TrainingAssistantProps {
  runId: string;
  metricsHistory: any[];
  currentEpoch: number;
  onApplyAction?: (action: string, value?: number) => void;
}

const TrainingAssistant: React.FC<TrainingAssistantProps> = ({
  runId,
  metricsHistory,
  currentEpoch,
  onApplyAction
}) => {
  const [analysis, setAnalysis] = useState<TrainingAnalysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [lastAnalyzedEpoch, setLastAnalyzedEpoch] = useState(0);
  const [insights, setInsights] = useState<string>('');

  // Auto-analyze every 5 epochs
  useEffect(() => {
    if (currentEpoch > 0 && currentEpoch % 5 === 0 && currentEpoch !== lastAnalyzedEpoch) {
      analyzeTraining();
    }
  }, [currentEpoch, lastAnalyzedEpoch]);

  const analyzeTraining = async () => {
    if (metricsHistory.length < 3) return;
    
    setLoading(true);
    try {
      const response = await axios.post(`/api/training/${runId}/analyze`, metricsHistory);
      setAnalysis(response.data);
      setLastAnalyzedEpoch(currentEpoch);
    } catch (error: any) {
      console.error('Analysis failed:', error);
      setAnalysis({
        status: 'analysis_failed',
        error: error.response?.data?.error || 'Analysis service unavailable',
        suggestions: ['Manual monitoring recommended']
      });
    }
    setLoading(false);
  };

  const fetchInsights = async () => {
    try {
      const response = await axios.get(`/api/training/${runId}/insights`);
      setInsights(response.data.insights);
    } catch (error) {
      console.error('Failed to fetch insights:', error);
    }
  };

  const handleApplyAction = async (action: any) => {
    if (!onApplyAction) return;
    
    try {
      await onApplyAction(action.action, action.value);
      
      // Re-analyze after applying action
      setTimeout(analyzeTraining, 2000);
    } catch (error) {
      console.error('Failed to apply action:', error);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircle color="success" />;
      case 'overfitting': return <Warning color="warning" />;
      case 'underfitting': return <TrendingUp color="info" />;
      case 'diverging': return <Error color="error" />;
      case 'plateau': return <Warning color="warning" />;
      case 'insufficient_data': return <Psychology color="disabled" />;
      case 'analysis_failed': return <Error color="error" />;
      default: return <Psychology />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'success';
      case 'overfitting': return 'warning';
      case 'underfitting': return 'info';
      case 'diverging': return 'error';
      case 'plateau': return 'warning';
      default: return 'default';
    }
  };

  const getStatusMessage = (status: string) => {
    switch (status) {
      case 'healthy': return 'Training is progressing well';
      case 'overfitting': return 'Model may be overfitting to training data';
      case 'underfitting': return 'Model may need more capacity or training';
      case 'diverging': return 'Training appears to be diverging';
      case 'plateau': return 'Training has plateaued';
      case 'insufficient_data': return 'Need more epochs for analysis';
      case 'analysis_failed': return 'Analysis service unavailable';
      default: return 'Unknown training status';
    }
  };

  return (
    <Box>
      {/* Main Analysis Card */}
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="h6" display="flex" alignItems="center">
              <Psychology sx={{ mr: 1 }} />
              AI Training Assistant
            </Typography>
            <Box>
              <Tooltip title="Refresh Analysis">
                <IconButton onClick={analyzeTraining} disabled={loading}>
                  <Refresh />
                </IconButton>
              </Tooltip>
              <Tooltip title="Get Training Insights">
                <IconButton onClick={fetchInsights}>
                  <Lightbulb />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>

          {loading && (
            <Box display="flex" alignItems="center" gap={2}>
              <CircularProgress size={20} />
              <Typography color="textSecondary">Analyzing training progress...</Typography>
            </Box>
          )}

          {analysis && !loading && (
            <Box>
              {/* Status */}
              <Box display="flex" alignItems="center" gap={2} mb={2}>
                {getStatusIcon(analysis.status)}
                <Chip
                  label={analysis.status.toUpperCase()}
                  color={getStatusColor(analysis.status) as any}
                  variant="outlined"
                />
                {analysis.confidence && (
                  <Typography variant="body2" color="textSecondary">
                    Confidence: {(analysis.confidence * 100).toFixed(0)}%
                  </Typography>
                )}
              </Box>

              <Typography variant="body2" color="textSecondary" mb={2}>
                {getStatusMessage(analysis.status)}
              </Typography>

              {/* Issues */}
              {analysis.issues && analysis.issues.length > 0 && (
                <Alert severity="warning" sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Issues Detected:
                  </Typography>
                  <List dense>
                    {analysis.issues.map((issue, index) => (
                      <ListItem key={index} sx={{ py: 0 }}>
                        <ListItemIcon sx={{ minWidth: 24 }}>
                          <Warning fontSize="small" />
                        </ListItemIcon>
                        <ListItemText primary={issue} />
                      </ListItem>
                    ))}
                  </List>
                </Alert>
              )}

              {/* Suggestions */}
              {analysis.suggestions && analysis.suggestions.length > 0 && (
                <Box mb={2}>
                  <Typography variant="subtitle2" gutterBottom>
                    Recommendations:
                  </Typography>
                  <List dense>
                    {analysis.suggestions.map((suggestion, index) => (
                      <ListItem key={index} sx={{ py: 0 }}>
                        <ListItemIcon sx={{ minWidth: 24 }}>
                          <Lightbulb fontSize="small" color="primary" />
                        </ListItemIcon>
                        <ListItemText primary={suggestion} />
                      </ListItem>
                    ))}
                  </List>
                </Box>
              )}

              {/* Auto Actions */}
              {analysis.auto_actions && analysis.auto_actions.length > 0 && (
                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Suggested Actions:
                  </Typography>
                  {analysis.auto_actions.map((action, index) => (
                    <Card key={index} variant="outlined" sx={{ mb: 1 }}>
                      <CardContent sx={{ py: 1, '&:last-child': { pb: 1 } }}>
                        <Box display="flex" justifyContent="space-between" alignItems="center">
                          <Box>
                            <Typography variant="body2" fontWeight="medium">
                              {action.action.replace('_', ' ').toUpperCase()}
                              {action.value && ` (${action.value})`}
                            </Typography>
                            <Typography variant="caption" color="textSecondary">
                              {action.reason}
                            </Typography>
                          </Box>
                          <Button
                            size="small"
                            variant="outlined"
                            startIcon={<AutoFixHigh />}
                            onClick={() => handleApplyAction(action)}
                            disabled={!onApplyAction}
                          >
                            Apply
                          </Button>
                        </Box>
                      </CardContent>
                    </Card>
                  ))}
                </Box>
              )}

              {/* Error */}
              {analysis.error && (
                <Alert severity="error">
                  {analysis.error}
                </Alert>
              )}
            </Box>
          )}

          {!analysis && !loading && metricsHistory.length < 3 && (
            <Alert severity="info">
              Training assistant will activate after 3 epochs of data are available.
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Training Insights */}
      {insights && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom display="flex" alignItems="center">
              <Lightbulb sx={{ mr: 1 }} />
              Training Insights
            </Typography>
            <Typography variant="body2" sx={{ fontStyle: 'italic' }}>
              {insights}
            </Typography>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default TrainingAssistant;