// src/components/ResourceMonitor.tsx
import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  LinearProgress,
  Chip,
  Alert
} from '@mui/material';
import {
  Memory,
  Storage,
  Speed,
  DeviceThermostat
} from '@mui/icons-material';
import { Line } from 'react-chartjs-2';

interface ResourceData {
  cpu_percent: number;
  memory_percent: number;
  memory_used_gb: number;
  memory_total_gb: number;
  disk_usage_percent: number;
  gpu_utilization?: number;
  gpu_memory_percent?: number;
  gpu_temperature?: number;
  timestamp: number;
}

interface ResourceMonitorProps {
  websocket: WebSocket | null;
}

const ResourceMonitor: React.FC<ResourceMonitorProps> = ({ websocket }) => {
  const [currentResources, setCurrentResources] = useState<ResourceData | null>(null);
  const [resourceHistory, setResourceHistory] = useState<ResourceData[]>([]);
  const [alerts, setAlerts] = useState<string[]>([]);

  useEffect(() => {
    if (!websocket) return;

    const handleMessage = (event: MessageEvent) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'resource_update') {
        const resources: ResourceData = data.resources;
        setCurrentResources(resources);
        
        // Keep last 50 data points for charts
        setResourceHistory(prev => {
          const updated = [...prev, resources];
          return updated.slice(-50);
        });
        
        // Check for alerts
        const newAlerts: string[] = [];
        if (resources.cpu_percent > 90) {
          newAlerts.push('High CPU usage detected');
        }
        if (resources.memory_percent > 90) {
          newAlerts.push('High memory usage detected');
        }
        if (resources.gpu_utilization && resources.gpu_utilization > 95) {
          newAlerts.push('GPU at maximum utilization');
        }
        if (resources.gpu_temperature && resources.gpu_temperature > 80) {
          newAlerts.push('High GPU temperature detected');
        }
        
        setAlerts(newAlerts);
      }
    };

    websocket.addEventListener('message', handleMessage);
    return () => websocket.removeEventListener('message', handleMessage);
  }, [websocket]);

  const getUsageColor = (percentage: number) => {
    if (percentage > 90) return 'error';
    if (percentage > 70) return 'warning';
    return 'primary';
  };

  const formatBytes = (bytes: number) => {
    return `${bytes.toFixed(1)} GB`;
  };

  // Chart data for resource usage over time
  const chartData = {
    labels: resourceHistory.map((_, index) => index.toString()),
    datasets: [
      {
        label: 'CPU %',
        data: resourceHistory.map(r => r.cpu_percent),
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        tension: 0.1
      },
      {
        label: 'Memory %',
        data: resourceHistory.map(r => r.memory_percent),
        borderColor: 'rgb(54, 162, 235)',
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        tension: 0.1
      },
      ...(resourceHistory.some(r => r.gpu_utilization !== undefined) ? [{
        label: 'GPU %',
        data: resourceHistory.map(r => r.gpu_utilization || 0),
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        tension: 0.1
      }] : [])
    ]
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      title: {
        display: true,
        text: 'Resource Usage Over Time'
      },
      legend: {
        position: 'top' as const,
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        title: {
          display: true,
          text: 'Usage %'
        }
      }
    }
  };

  if (!currentResources) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            System Resources
          </Typography>
          <Typography color="textSecondary">
            Waiting for resource data...
          </Typography>
        </CardContent>
      </Card>
    );
  }

  return (
    <Box>
      {/* Alerts */}
      {alerts.length > 0 && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          {alerts.join(', ')}
        </Alert>
      )}

      {/* Current Resource Usage */}
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            System Resources
          </Typography>
          
          <Grid container spacing={3}>
            {/* CPU */}
            <Grid item xs={12} sm={6} md={3}>
              <Box display="flex" alignItems="center" mb={1}>
                <Speed sx={{ mr: 1 }} />
                <Typography variant="subtitle2">CPU</Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={currentResources.cpu_percent}
                color={getUsageColor(currentResources.cpu_percent) as any}
                sx={{ height: 8, borderRadius: 4, mb: 1 }}
              />
              <Typography variant="body2" color="textSecondary">
                {currentResources.cpu_percent.toFixed(1)}%
              </Typography>
            </Grid>

            {/* Memory */}
            <Grid item xs={12} sm={6} md={3}>
              <Box display="flex" alignItems="center" mb={1}>
                <Memory sx={{ mr: 1 }} />
                <Typography variant="subtitle2">Memory</Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={currentResources.memory_percent}
                color={getUsageColor(currentResources.memory_percent) as any}
                sx={{ height: 8, borderRadius: 4, mb: 1 }}
              />
              <Typography variant="body2" color="textSecondary">
                {formatBytes(currentResources.memory_used_gb)} / {formatBytes(currentResources.memory_total_gb)}
              </Typography>
            </Grid>

            {/* Disk */}
            <Grid item xs={12} sm={6} md={3}>
              <Box display="flex" alignItems="center" mb={1}>
                <Storage sx={{ mr: 1 }} />
                <Typography variant="subtitle2">Disk</Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={currentResources.disk_usage_percent}
                color={getUsageColor(currentResources.disk_usage_percent) as any}
                sx={{ height: 8, borderRadius: 4, mb: 1 }}
              />
              <Typography variant="body2" color="textSecondary">
                {currentResources.disk_usage_percent.toFixed(1)}%
              </Typography>
            </Grid>

            {/* GPU (if available) */}
            {currentResources.gpu_utilization !== undefined && (
              <Grid item xs={12} sm={6} md={3}>
                <Box display="flex" alignItems="center" mb={1}>
                  <DeviceThermostat sx={{ mr: 1 }} />
                  <Typography variant="subtitle2">GPU</Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={currentResources.gpu_utilization}
                  color={getUsageColor(currentResources.gpu_utilization) as any}
                  sx={{ height: 8, borderRadius: 4, mb: 1 }}
                />
                <Box display="flex" gap={1}>
                  <Chip 
                    label={`${currentResources.gpu_utilization.toFixed(1)}%`} 
                    size="small" 
                  />
                  {currentResources.gpu_memory_percent && (
                    <Chip 
                      label={`Mem: ${currentResources.gpu_memory_percent.toFixed(1)}%`} 
                      size="small" 
                      variant="outlined"
                    />
                  )}
                  {currentResources.gpu_temperature && (
                    <Chip 
                      label={`${currentResources.gpu_temperature}°C`} 
                      size="small" 
                      color={currentResources.gpu_temperature > 80 ? 'error' : 'default'}
                    />
                  )}
                </Box>
              </Grid>
            )}
          </Grid>
        </CardContent>
      </Card>

      {/* Resource Usage Chart */}
      {resourceHistory.length > 5 && (
        <Card>
          <CardContent>
            <Line data={chartData} options={chartOptions} />
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default ResourceMonitor;