// src/components/TrainingPage.tsx
import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  Alert,
  Tabs,
  Tab,
  Divider
} from '@mui/material';
import { useSearchParams } from 'react-router-dom';
import TrainingControlPanel from './TrainingControlPanel';
import MetricsDashboard from './MetricsDashboard';
import ResourceMonitor from './ResourceMonitor';
import TrainingAssistant from './TrainingAssistant';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`training-tabpanel-${index}`}
      aria-labelledby={`training-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const TrainingPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const runId = searchParams.get('runId');
  const [websocket, setWebsocket] = useState<WebSocket | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('connecting');
  const [trainingStatus, setTrainingStatus] = useState<string>('unknown');
  const [tabValue, setTabValue] = useState(0);
  const [metricsHistory, setMetricsHistory] = useState<any[]>([]);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!runId) return;

    // Establish WebSocket connection
    const connectWebSocket = () => {
      const ws = new WebSocket(`ws://localhost:8000/ws/${runId}`);
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        setConnectionStatus('connected');
        setWebsocket(ws);
        wsRef.current = ws;
      };

      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setConnectionStatus('disconnected');
        setWebsocket(null);
        wsRef.current = null;
        
        // Attempt to reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000);
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('disconnected');
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('WebSocket message:', data);
        
        // Handle status updates
        if (data.event === 'training_paused') {
          setTrainingStatus('paused');
        } else if (data.event === 'training_resumed') {
          setTrainingStatus('running');
        } else if (data.event === 'training_stopped') {
          setTrainingStatus('stopped');
        } else if (data.event === 'training_completed') {
          setTrainingStatus('completed');
        }
        
        // Handle metrics updates for AI assistant
        if (data.type === 'training_update' || data.type === 'evaluation_update') {
          setCurrentEpoch(data.epoch || 0);
          if (data.metrics) {
            setMetricsHistory(prev => {
              const updated = [...prev, data.metrics];
              return updated.slice(-50); // Keep last 50 metrics
            });
          }
        }
      };
    };

    connectWebSocket();

    // Cleanup on unmount
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [runId]);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const getConnectionStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return 'success';
      case 'connecting': return 'info';
      case 'disconnected': return 'error';
      default: return 'info';
    }
  };

  const getConnectionStatusText = () => {
    switch (connectionStatus) {
      case 'connected': return 'Connected to training session';
      case 'connecting': return 'Connecting to training session...';
      case 'disconnected': return 'Disconnected from training session. Attempting to reconnect...';
      default: return 'Unknown connection status';
    }
  };

  const handleApplyAction = async (action: string, value?: number) => {
    try {
      const params: any = {};
      
      if (action === 'reduce_lr' && value) {
        params.learning_rate = value;
      } else if (action === 'gradient_clip' && value) {
        params.gradient_clip = value;
      }
      
      const response = await fetch(`/api/training/${runId}/update-params`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      });
      
      if (!response.ok) {
        throw new Error('Failed to apply action');
      }
      
      console.log('Action applied successfully:', action, value);
    } catch (error) {
      console.error('Failed to apply action:', error);
      throw error;
    }
  };

  if (!runId) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4 }}>
        <Alert severity="error">
          No training run ID provided. Please start a training session first.
        </Alert>
      </Container>
    );
  }

  return (
    <Container maxWidth="xl" sx={{ mt: 2, mb: 4 }}>
      {/* Header */}
      <Box mb={3}>
        <Typography variant="h4" component="h1" gutterBottom>
          Training Session
        </Typography>
        <Typography variant="subtitle1" color="textSecondary" gutterBottom>
          Run ID: {runId}
        </Typography>
        
        <Alert 
          severity={getConnectionStatusColor() as any} 
          sx={{ mt: 2 }}
        >
          {getConnectionStatusText()}
        </Alert>
      </Box>

      {/* Main Content */}
      <Grid container spacing={3}>
        {/* Left Column - Control Panel */}
        <Grid item xs={12} lg={4}>
          <TrainingControlPanel 
            runId={runId} 
            onStatusChange={setTrainingStatus}
          />
        </Grid>

        {/* Right Column - Metrics and Visualizations */}
        <Grid item xs={12} lg={8}>
          <Card>
            <CardContent>
              <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                <Tabs value={tabValue} onChange={handleTabChange} aria-label="training tabs">
                  <Tab label="Metrics Dashboard" />
                  <Tab label="AI Assistant" />
                  <Tab label="System Resources" />
                  <Tab label="Training Logs" />
                </Tabs>
              </Box>
              
              <TabPanel value={tabValue} index={0}>
                <MetricsDashboard runId={runId} websocket={websocket} />
              </TabPanel>
              
              <TabPanel value={tabValue} index={1}>
                <TrainingAssistant 
                  runId={runId} 
                  metricsHistory={metricsHistory}
                  currentEpoch={currentEpoch}
                  onApplyAction={handleApplyAction}
                />
              </TabPanel>
              
              <TabPanel value={tabValue} index={2}>
                <ResourceMonitor websocket={websocket} />
              </TabPanel>
              
              <TabPanel value={tabValue} index={3}>
                <TrainingLogs runId={runId} websocket={websocket} />
              </TabPanel>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
};

// Training Logs Component
const TrainingLogs: React.FC<{ runId: string; websocket: WebSocket | null }> = ({ 
  runId, 
  websocket 
}) => {
  const [logs, setLogs] = useState<Array<{ timestamp: string; message: string; type: string }>>([]);
  const logsEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!websocket) return;

    const handleMessage = (event: MessageEvent) => {
      const data = JSON.parse(event.data);
      
      if (data.event === 'log' || data.event === 'error') {
        const newLog = {
          timestamp: new Date().toLocaleTimeString(),
          message: data.message,
          type: data.event
        };
        setLogs(prev => [...prev, newLog]);
      }
    };

    websocket.addEventListener('message', handleMessage);
    return () => websocket.removeEventListener('message', handleMessage);
  }, [websocket]);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Training Logs
      </Typography>
      <Box
        sx={{
          height: 400,
          overflow: 'auto',
          bgcolor: 'grey.100',
          p: 2,
          borderRadius: 1,
          fontFamily: 'monospace'
        }}
      >
        {logs.map((log, index) => (
          <Box key={index} sx={{ mb: 1 }}>
            <Typography
              variant="body2"
              component="div"
              sx={{
                color: log.type === 'error' ? 'error.main' : 'text.primary',
                fontSize: '0.875rem'
              }}
            >
              <span style={{ color: 'grey' }}>[{log.timestamp}]</span> {log.message}
            </Typography>
          </Box>
        ))}
        <div ref={logsEndRef} />
      </Box>
    </Box>
  );
};

// Model Info Component
const ModelInfo: React.FC<{ runId: string }> = ({ runId }) => {
  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Model Information
      </Typography>
      <Typography variant="body2" color="textSecondary">
        Model details and configuration will be displayed here.
      </Typography>
      {/* TODO: Add model configuration display */}
    </Box>
  );
};

export default TrainingPage;