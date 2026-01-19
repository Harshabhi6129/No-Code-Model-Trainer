// src/components/MetricsDashboard.tsx
import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  LinearProgress
} from '@mui/material';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface MetricsData {
  epoch: number;
  batch: number;
  total_batches: number;
  metrics: {
    train_loss: number;
    train_accuracy: number;
    val_loss?: number;
    val_accuracy?: number;
    learning_rate: number;
    grad_norm?: number;
    train_f1?: number;
    val_f1?: number;
  };
  time_elapsed: number;
  estimated_time_remaining?: number;
}

interface MetricsDashboardProps {
  runId: string;
  websocket: WebSocket | null;
}

const MetricsDashboard: React.FC<MetricsDashboardProps> = ({ runId, websocket }) => {
  const [metricsHistory, setMetricsHistory] = useState<MetricsData[]>([]);
  const [currentMetrics, setCurrentMetrics] = useState<MetricsData | null>(null);
  const [parameterChanges, setParameterChanges] = useState<Array<{epoch: number, params: any}>>([]);

  useEffect(() => {
    if (!websocket) return;

    const handleMessage = (event: MessageEvent) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'training_update') {
        const newMetric: MetricsData = {
          epoch: data.epoch,
          batch: data.batch,
          total_batches: data.total_batches,
          metrics: data.metrics,
          time_elapsed: data.time_elapsed,
          estimated_time_remaining: data.estimated_time_remaining
        };
        
        setCurrentMetrics(newMetric);
        setMetricsHistory(prev => [...prev, newMetric]);
      }
      
      if (data.event === 'params_updated') {
        setParameterChanges(prev => [...prev, {
          epoch: currentMetrics?.epoch || 0,
          params: data.params
        }]);
      }
    };

    websocket.addEventListener('message', handleMessage);
    return () => websocket.removeEventListener('message', handleMessage);
  }, [websocket, currentMetrics?.epoch]);

  // Chart configurations
  const lossChartOptions: ChartOptions<'line'> = {
    responsive: true,
    interaction: {
      mode: 'index' as const,
      intersect: false,
    },
    plugins: {
      title: {
        display: true,
        text: 'Training & Validation Loss'
      },
      legend: {
        position: 'top' as const,
      },
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Epoch'
        }
      },
      y: {
        display: true,
        title: {
          display: true,
          text: 'Loss'
        }
      }
    }
  };

  const accuracyChartOptions: ChartOptions<'line'> = {
    responsive: true,
    plugins: {
      title: {
        display: true,
        text: 'Training & Validation Accuracy'
      },
      legend: {
        position: 'top' as const,
      },
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Epoch'
        }
      },
      y: {
        display: true,
        title: {
          display: true,
          text: 'Accuracy'
        },
        min: 0,
        max: 1
      }
    }
  };

  const learningRateChartOptions: ChartOptions<'line'> = {
    responsive: true,
    plugins: {
      title: {
        display: true,
        text: 'Learning Rate Schedule'
      },
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Epoch'
        }
      },
      y: {
        display: true,
        title: {
          display: true,
          text: 'Learning Rate'
        },
        type: 'logarithmic'
      }
    }
  };

  // Prepare chart data
  const epochs = metricsHistory.map(m => m.epoch);
  
  const lossChartData = {
    labels: epochs,
    datasets: [
      {
        label: 'Training Loss',
        data: metricsHistory.map(m => m.metrics.train_loss),
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        tension: 0.1
      },
      ...(metricsHistory.some(m => m.metrics.val_loss !== undefined) ? [{
        label: 'Validation Loss',
        data: metricsHistory.map(m => m.metrics.val_loss || null),
        borderColor: 'rgb(54, 162, 235)',
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        tension: 0.1
      }] : [])
    ]
  };

  const accuracyChartData = {
    labels: epochs,
    datasets: [
      {
        label: 'Training Accuracy',
        data: metricsHistory.map(m => m.metrics.train_accuracy),
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        tension: 0.1
      },
      ...(metricsHistory.some(m => m.metrics.val_accuracy !== undefined) ? [{
        label: 'Validation Accuracy',
        data: metricsHistory.map(m => m.metrics.val_accuracy || null),
        borderColor: 'rgb(153, 102, 255)',
        backgroundColor: 'rgba(153, 102, 255, 0.2)',
        tension: 0.1
      }] : [])
    ]
  };

  const learningRateChartData = {
    labels: epochs,
    datasets: [
      {
        label: 'Learning Rate',
        data: metricsHistory.map(m => m.metrics.learning_rate),
        borderColor: 'rgb(255, 159, 64)',
        backgroundColor: 'rgba(255, 159, 64, 0.2)',
        tension: 0.1
      }
    ]
  };

  const formatTime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const getLatestMetrics = () => {
    if (!currentMetrics) return null;
    return [
      { name: 'Train Loss', value: currentMetrics.metrics.train_loss?.toFixed(4) || 'N/A' },
      { name: 'Train Accuracy', value: currentMetrics.metrics.train_accuracy?.toFixed(3) || 'N/A' },
      { name: 'Val Loss', value: currentMetrics.metrics.val_loss?.toFixed(4) || 'N/A' },
      { name: 'Val Accuracy', value: currentMetrics.metrics.val_accuracy?.toFixed(3) || 'N/A' },
      { name: 'Learning Rate', value: currentMetrics.metrics.learning_rate?.toExponential(2) || 'N/A' },
      { name: 'Grad Norm', value: currentMetrics.metrics.grad_norm?.toFixed(3) || 'N/A' },
    ];
  };

  return (
    <Box>
      {/* Current Progress */}
      {currentMetrics && (
        <Card sx={{ mb: 2 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Current Progress
            </Typography>
            <Grid container spacing={2} alignItems="center">
              <Grid item xs={12} md={8}>
                <Typography variant="body2" color="textSecondary" gutterBottom>
                  Epoch {currentMetrics.epoch} - Batch {currentMetrics.batch}/{currentMetrics.total_batches}
                </Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={(currentMetrics.batch / currentMetrics.total_batches) * 100}
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Grid>
              <Grid item xs={12} md={4}>
                <Typography variant="body2" color="textSecondary">
                  Elapsed: {formatTime(currentMetrics.time_elapsed)}
                </Typography>
                {currentMetrics.estimated_time_remaining && (
                  <Typography variant="body2" color="textSecondary">
                    Remaining: {formatTime(currentMetrics.estimated_time_remaining)}
                  </Typography>
                )}
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      )}

      {/* Current Metrics Table */}
      {currentMetrics && (
        <Card sx={{ mb: 2 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Latest Metrics
            </Typography>
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Metric</TableCell>
                    <TableCell align="right">Value</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {getLatestMetrics()?.map((metric) => (
                    <TableRow key={metric.name}>
                      <TableCell component="th" scope="row">
                        {metric.name}
                      </TableCell>
                      <TableCell align="right">
                        <Chip label={metric.value} size="small" />
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      )}

      {/* Charts */}
      {metricsHistory.length > 0 && (
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Line data={lossChartData} options={lossChartOptions} />
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Line data={accuracyChartData} options={accuracyChartOptions} />
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Line data={learningRateChartData} options={learningRateChartOptions} />
              </CardContent>
            </Card>
          </Grid>

          {/* Parameter Changes History */}
          {parameterChanges.length > 0 && (
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Parameter Changes
                  </Typography>
                  <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 300 }}>
                    <Table size="small" stickyHeader>
                      <TableHead>
                        <TableRow>
                          <TableCell>Epoch</TableCell>
                          <TableCell>Parameter</TableCell>
                          <TableCell align="right">New Value</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {parameterChanges.flatMap((change) =>
                          Object.entries(change.params).map(([param, value]) => (
                            <TableRow key={`${change.epoch}-${param}`}>
                              <TableCell>{change.epoch}</TableCell>
                              <TableCell>{param.replace('_', ' ')}</TableCell>
                              <TableCell align="right">
                                <Chip 
                                  label={typeof value === 'number' ? value.toExponential(2) : value} 
                                  size="small" 
                                  color="primary"
                                />
                              </TableCell>
                            </TableRow>
                          ))
                        )}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </CardContent>
              </Card>
            </Grid>
          )}
        </Grid>
      )}
    </Box>
  );
};

export default MetricsDashboard;