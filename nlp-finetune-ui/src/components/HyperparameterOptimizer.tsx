// src/components/HyperparameterOptimizer.tsx
import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Grid,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  List,
  ListItem,
  ListItemText,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper
} from '@mui/material';
import {
  TuneOutlined,
  PlayArrow,
  Stop,
  AutoAwesome
} from '@mui/icons-material';
import axios from 'axios';

interface SearchSpace {
  [key: string]: number[];
}

interface OptimizationResult {
  trial: number;
  total_trials: number;
  current_score: number;
  best_score: number;
  best_config: any;
}

interface HyperparameterOptimizerProps {
  baseConfig: any;
  onOptimizationComplete?: (bestConfig: any) => void;
}

const HyperparameterOptimizer: React.FC<HyperparameterOptimizerProps> = ({
  baseConfig,
  onOptimizationComplete
}) => {
  const [open, setOpen] = useState(false);
  const [searchSpace, setSearchSpace] = useState<SearchSpace>({});
  const [searchType, setSearchType] = useState<'random' | 'grid'>('random');
  const [maxTrials, setMaxTrials] = useState(10);
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState<OptimizationResult | null>(null);
  const [predefinedSpaces, setPredefinedSpaces] = useState<any>({});
  const [selectedSpace, setSelectedSpace] = useState('');

  useEffect(() => {
    fetchPredefinedSpaces();
  }, []);

  const fetchPredefinedSpaces = async () => {
    try {
      const response = await axios.get('/api/hyperopt/search-spaces');
      setPredefinedSpaces(response.data.search_spaces);
    } catch (error) {
      console.error('Failed to fetch search spaces:', error);
    }
  };

  const handleSpaceSelection = (spaceKey: string) => {
    setSelectedSpace(spaceKey);
    setSearchSpace(predefinedSpaces[spaceKey] || {});
  };

  const addParameter = () => {
    const paramName = prompt('Enter parameter name:');
    if (paramName && !searchSpace[paramName]) {
      setSearchSpace(prev => ({
        ...prev,
        [paramName]: [0.001, 0.01, 0.1]
      }));
    }
  };

  const updateParameterValues = (paramName: string, values: string) => {
    try {
      const parsedValues = values.split(',').map(v => parseFloat(v.trim()));
      setSearchSpace(prev => ({
        ...prev,
        [paramName]: parsedValues
      }));
    } catch (error) {
      console.error('Invalid values format');
    }
  };

  const removeParameter = (paramName: string) => {
    setSearchSpace(prev => {
      const newSpace = { ...prev };
      delete newSpace[paramName];
      return newSpace;
    });
  };

  const startOptimization = async () => {
    if (Object.keys(searchSpace).length === 0) {
      alert('Please define at least one parameter to optimize');
      return;
    }

    setRunning(true);
    setResults(null);

    try {
      const payload = {
        base_config: baseConfig,
        search_space: searchSpace,
        search_type: searchType,
        max_trials: maxTrials
      };

      const response = await axios.post('/api/hyperopt/start', payload);
      const runId = response.data.run_id;

      // Set up WebSocket to listen for progress
      const ws = new WebSocket(`ws://localhost:8000/ws/${runId}`);
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.event === 'hyperopt_progress') {
          setResults(data);
        } else if (data.event === 'hyperopt_completed') {
          setResults(data);
          setRunning(false);
          onOptimizationComplete?.(data.best_config);
          ws.close();
        } else if (data.event === 'hyperopt_error') {
          console.error('Optimization error:', data.error);
          setRunning(false);
          ws.close();
        }
      };

      ws.onerror = () => {
        setRunning(false);
      };

    } catch (error) {
      console.error('Failed to start optimization:', error);
      setRunning(false);
    }
  };

  const estimatedTrials = () => {
    const combinations = Object.values(searchSpace).reduce((acc, values) => acc * values.length, 1);
    return searchType === 'grid' ? Math.min(combinations, maxTrials) : maxTrials;
  };

  return (
    <Box>
      <Button
        variant="contained"
        startIcon={<TuneOutlined />}
        onClick={() => setOpen(true)}
        sx={{ mb: 2 }}
      >
        Optimize Hyperparameters
      </Button>

      <Dialog open={open} onClose={() => setOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>
          <Box display="flex" alignItems="center">
            <AutoAwesome sx={{ mr: 1 }} />
            Hyperparameter Optimization
          </Box>
        </DialogTitle>

        <DialogContent>
          {/* Predefined Spaces */}
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Quick Start Templates
              </Typography>
              <Grid container spacing={1}>
                {Object.keys(predefinedSpaces).map((spaceKey) => (
                  <Grid item key={spaceKey}>
                    <Chip
                      label={spaceKey.replace('_', ' ')}
                      onClick={() => handleSpaceSelection(spaceKey)}
                      color={selectedSpace === spaceKey ? 'primary' : 'default'}
                      variant={selectedSpace === spaceKey ? 'filled' : 'outlined'}
                    />
                  </Grid>
                ))}
              </Grid>
            </CardContent>
          </Card>

          {/* Search Configuration */}
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Search Configuration
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <FormControl fullWidth>
                    <InputLabel>Search Type</InputLabel>
                    <Select
                      value={searchType}
                      onChange={(e) => setSearchType(e.target.value as 'random' | 'grid')}
                    >
                      <MenuItem value="random">Random Search</MenuItem>
                      <MenuItem value="grid">Grid Search</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    label="Max Trials"
                    type="number"
                    value={maxTrials}
                    onChange={(e) => setMaxTrials(parseInt(e.target.value))}
                    inputProps={{ min: 1, max: 100 }}
                  />
                </Grid>
              </Grid>
              <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
                Estimated trials: {estimatedTrials()}
              </Typography>
            </CardContent>
          </Card>

          {/* Search Space Definition */}
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                <Typography variant="h6">
                  Search Space
                </Typography>
                <Button size="small" onClick={addParameter}>
                  Add Parameter
                </Button>
              </Box>

              {Object.keys(searchSpace).length === 0 ? (
                <Typography color="textSecondary">
                  No parameters defined. Use a template or add parameters manually.
                </Typography>
              ) : (
                <List>
                  {Object.entries(searchSpace).map(([paramName, values]) => (
                    <ListItem key={paramName}>
                      <ListItemText
                        primary={paramName}
                        secondary={
                          <TextField
                            fullWidth
                            size="small"
                            value={values.join(', ')}
                            onChange={(e) => updateParameterValues(paramName, e.target.value)}
                            placeholder="Enter comma-separated values"
                            sx={{ mt: 1 }}
                          />
                        }
                      />
                      <Button
                        size="small"
                        color="error"
                        onClick={() => removeParameter(paramName)}
                      >
                        Remove
                      </Button>
                    </ListItem>
                  ))}
                </List>
              )}
            </CardContent>
          </Card>

          {/* Progress */}
          {running && results && (
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Optimization Progress
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={(results.trial / results.total_trials) * 100}
                  sx={{ mb: 2 }}
                />
                <Typography variant="body2" gutterBottom>
                  Trial {results.trial} of {results.total_trials}
                </Typography>
                <Typography variant="body2">
                  Current Score: {results.current_score.toFixed(4)} | 
                  Best Score: {results.best_score.toFixed(4)}
                </Typography>
              </CardContent>
            </Card>
          )}

          {/* Results */}
          {results && !running && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Optimization Results
                </Typography>
                <Typography variant="body1" gutterBottom>
                  Best Score: <strong>{results.best_score.toFixed(4)}</strong>
                </Typography>
                <Typography variant="subtitle2" gutterBottom>
                  Best Configuration:
                </Typography>
                <TableContainer component={Paper} variant="outlined">
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Parameter</TableCell>
                        <TableCell>Value</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {Object.entries(results.best_config).map(([key, value]) => (
                        <TableRow key={key}>
                          <TableCell>{key}</TableCell>
                          <TableCell>{String(value)}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          )}
        </DialogContent>

        <DialogActions>
          <Button onClick={() => setOpen(false)}>
            Close
          </Button>
          <Button
            variant="contained"
            startIcon={running ? <Stop /> : <PlayArrow />}
            onClick={startOptimization}
            disabled={running || Object.keys(searchSpace).length === 0}
          >
            {running ? 'Running...' : 'Start Optimization'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default HyperparameterOptimizer;