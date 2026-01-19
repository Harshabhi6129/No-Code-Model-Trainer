// src/components/TrainingReportViewer.tsx
import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Grid,
  Chip,
  List,
  ListItem,
  ListItemText,
  Divider,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  CircularProgress
} from '@mui/material';
import {
  Download,
  Assessment,
  Share,
  Print,
  Visibility
} from '@mui/icons-material';
import axios from 'axios';

interface TrainingReport {
  run_id: string;
  generated_at: string;
  config: any;
  summary: {
    total_epochs: number;
    best_val_acc?: number;
    best_val_loss?: number;
    total_time: number;
    train_loss_improvement?: number;
    val_loss_improvement?: number;
  };
  insights: string[];
  charts: {
    loss_curve?: string;
    accuracy_curve?: string;
    learning_rate?: string;
  };
  metrics_count: number;
}

interface TrainingReportViewerProps {
  runId: string;
  open: boolean;
  onClose: () => void;
}

const TrainingReportViewer: React.FC<TrainingReportViewerProps> = ({
  runId,
  open,
  onClose
}) => {
  const [report, setReport] = useState<TrainingReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open && runId) {
      generateReport();
    }
  }, [open, runId]);

  const generateReport = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.get(`/api/training/${runId}/report`);
      setReport(response.data);
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to generate report');
    }
    
    setLoading(false);
  };

  const downloadModelPackage = async () => {
    try {
      const response = await axios.get(`/api/training/${runId}/export-package`, {
        responseType: 'blob'
      });
      
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `${runId}_complete_package.zip`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Failed to download model package:', error);
    }
  };

  const formatTime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  const formatDate = (isoString: string) => {
    return new Date(isoString).toLocaleString();
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Typography variant="h6">Training Report</Typography>
          <Box>
            <Button
              startIcon={<Download />}
              onClick={downloadModelPackage}
              disabled={!report}
              sx={{ mr: 1 }}
            >
              Download Package
            </Button>
            <Button startIcon={<Print />} disabled={!report}>
              Print Report
            </Button>
          </Box>
        </Box>
      </DialogTitle>
      
      <DialogContent>
        {loading && (
          <Box display="flex" justifyContent="center" p={4}>
            <CircularProgress />
            <Typography sx={{ ml: 2 }}>Generating report...</Typography>
          </Box>
        )}

        {error && (
          <Box p={2}>
            <Typography color="error">{error}</Typography>
            <Button onClick={generateReport} sx={{ mt: 2 }}>
              Retry
            </Button>
          </Box>
        )}

        {report && (
          <Box>
            {/* Header Info */}
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <Typography variant="h6" gutterBottom>
                      Run Information
                    </Typography>
                    <Typography><strong>Run ID:</strong> {report.run_id}</Typography>
                    <Typography><strong>Generated:</strong> {formatDate(report.generated_at)}</Typography>
                    <Typography><strong>Total Epochs:</strong> {report.summary.total_epochs}</Typography>
                    <Typography><strong>Training Time:</strong> {formatTime(report.summary.total_time)}</Typography>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <Typography variant="h6" gutterBottom>
                      Best Results
                    </Typography>
                    {report.summary.best_val_acc && (
                      <Chip 
                        label={`Best Accuracy: ${(report.summary.best_val_acc * 100).toFixed(2)}%`}
                        color="success"
                        sx={{ mr: 1, mb: 1 }}
                      />
                    )}
                    {report.summary.best_val_loss && (
                      <Chip 
                        label={`Best Loss: ${report.summary.best_val_loss.toFixed(4)}`}
                        color="primary"
                        sx={{ mr: 1, mb: 1 }}
                      />
                    )}
                  </Grid>
                </Grid>
              </CardContent>
            </Card>

            {/* Training Configuration */}
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Training Configuration
                </Typography>
                <TableContainer component={Paper} variant="outlined">
                  <Table size="small">
                    <TableBody>
                      {Object.entries(report.config).map(([key, value]) => (
                        <TableRow key={key}>
                          <TableCell component="th" scope="row" sx={{ fontWeight: 'bold' }}>
                            {key.replace('_', ' ').toUpperCase()}
                          </TableCell>
                          <TableCell>
                            {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>

            {/* Training Insights */}
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Training Insights
                </Typography>
                <List>
                  {report.insights.map((insight, index) => (
                    <ListItem key={index}>
                      <ListItemText primary={insight} />
                    </ListItem>
                  ))}
                </List>
              </CardContent>
            </Card>

            {/* Charts */}
            {Object.keys(report.charts).length > 0 && (
              <Card sx={{ mb: 3 }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Training Visualizations
                  </Typography>
                  <Grid container spacing={2}>
                    {report.charts.loss_curve && (
                      <Grid item xs={12} md={6}>
                        <Paper variant="outlined" sx={{ p: 1 }}>
                          <Typography variant="subtitle2" gutterBottom>
                            Loss Curves
                          </Typography>
                          <img
                            src={`data:image/png;base64,${report.charts.loss_curve}`}
                            alt="Loss Curves"
                            style={{ width: '100%', height: 'auto' }}
                          />
                        </Paper>
                      </Grid>
                    )}
                    
                    {report.charts.accuracy_curve && (
                      <Grid item xs={12} md={6}>
                        <Paper variant="outlined" sx={{ p: 1 }}>
                          <Typography variant="subtitle2" gutterBottom>
                            Accuracy Curves
                          </Typography>
                          <img
                            src={`data:image/png;base64,${report.charts.accuracy_curve}`}
                            alt="Accuracy Curves"
                            style={{ width: '100%', height: 'auto' }}
                          />
                        </Paper>
                      </Grid>
                    )}
                    
                    {report.charts.learning_rate && (
                      <Grid item xs={12} md={6}>
                        <Paper variant="outlined" sx={{ p: 1 }}>
                          <Typography variant="subtitle2" gutterBottom>
                            Learning Rate Schedule
                          </Typography>
                          <img
                            src={`data:image/png;base64,${report.charts.learning_rate}`}
                            alt="Learning Rate Schedule"
                            style={{ width: '100%', height: 'auto' }}
                          />
                        </Paper>
                      </Grid>
                    )}
                  </Grid>
                </CardContent>
              </Card>
            )}

            {/* Summary Statistics */}
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Performance Summary
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={6} md={3}>
                    <Paper variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h4" color="primary">
                        {report.summary.total_epochs}
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        Total Epochs
                      </Typography>
                    </Paper>
                  </Grid>
                  
                  {report.summary.best_val_acc && (
                    <Grid item xs={6} md={3}>
                      <Paper variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
                        <Typography variant="h4" color="success.main">
                          {(report.summary.best_val_acc * 100).toFixed(1)}%
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          Best Accuracy
                        </Typography>
                      </Paper>
                    </Grid>
                  )}
                  
                  <Grid item xs={6} md={3}>
                    <Paper variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h4" color="info.main">
                        {formatTime(report.summary.total_time)}
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        Training Time
                      </Typography>
                    </Paper>
                  </Grid>
                  
                  <Grid item xs={6} md={3}>
                    <Paper variant="outlined" sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h4" color="warning.main">
                        {report.metrics_count}
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        Data Points
                      </Typography>
                    </Paper>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Box>
        )}
      </DialogContent>
      
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
};

export default TrainingReportViewer;