// src/components/TrainingRunsManager.tsx
import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Grid,
  Tooltip
} from '@mui/material';
import {
  Visibility,
  Download,
  Delete,
  Assessment,
  Compare,
  Refresh
} from '@mui/icons-material';
import axios from 'axios';
import TrainingReportViewer from './TrainingReportViewer';

interface TrainingRun {
  run_id: string;
  created_at: string;
  has_metrics: boolean;
  has_model: boolean;
  total_epochs?: number;
  final_loss?: number;
  best_val_acc?: number;
}

const TrainingRunsManager: React.FC = () => {
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [reportOpen, setReportOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [runToDelete, setRunToDelete] = useState<string | null>(null);
  const [selectedForComparison, setSelectedForComparison] = useState<string[]>([]);

  useEffect(() => {
    fetchRuns();
  }, []);

  const fetchRuns = async () => {
    setLoading(true);
    try {
      const response = await axios.get('/api/training/list');
      setRuns(response.data);
    } catch (error) {
      console.error('Failed to fetch training runs:', error);
    }
    setLoading(false);
  };

  const handleViewReport = (runId: string) => {
    setSelectedRun(runId);
    setReportOpen(true);
  };

  const handleDownloadModel = async (runId: string) => {
    try {
      const response = await axios.get(`/export/${runId}`, {
        responseType: 'blob'
      });
      
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `${runId}.zip`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Failed to download model:', error);
    }
  };

  const handleDeleteRun = async () => {
    if (!runToDelete) return;

    try {
      await axios.delete(`/api/training/${runToDelete}`);
      setRuns(prev => prev.filter(run => run.run_id !== runToDelete));
      setDeleteDialogOpen(false);
      setRunToDelete(null);
    } catch (error) {
      console.error('Failed to delete run:', error);
    }
  };

  const toggleComparison = (runId: string) => {
    setSelectedForComparison(prev => {
      if (prev.includes(runId)) {
        return prev.filter(id => id !== runId);
      } else if (prev.length < 3) {
        return [...prev, runId];
      } else {
        return prev;
      }
    });
  };

  const formatDate = (isoString: string) => {
    return new Date(isoString).toLocaleDateString();
  };

  const getStatusChip = (run: TrainingRun) => {
    if (run.has_model) {
      return <Chip label="Completed" color="success" size="small" />;
    } else if (run.has_metrics) {
      return <Chip label="Training" color="warning" size="small" />;
    } else {
      return <Chip label="Failed" color="error" size="small" />;
    }
  };

  return (
    <Box>
      <Card>
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="h6">
              Training Runs Manager
            </Typography>
            <Box>
              {selectedForComparison.length > 1 && (
                <Button
                  startIcon={<Compare />}
                  sx={{ mr: 1 }}
                  onClick={() => {/* TODO: Implement comparison */}}
                >
                  Compare ({selectedForComparison.length})
                </Button>
              )}
              <Button
                startIcon={<Refresh />}
                onClick={fetchRuns}
                disabled={loading}
              >
                Refresh
              </Button>
            </Box>
          </Box>

          <TableContainer component={Paper} variant="outlined">
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell padding="checkbox">
                    {/* Comparison checkbox header */}
                  </TableCell>
                  <TableCell>Run ID</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Created</TableCell>
                  <TableCell>Epochs</TableCell>
                  <TableCell>Best Accuracy</TableCell>
                  <TableCell>Final Loss</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {runs.map((run) => (
                  <TableRow key={run.run_id}>
                    <TableCell padding="checkbox">
                      <input
                        type="checkbox"
                        checked={selectedForComparison.includes(run.run_id)}
                        onChange={() => toggleComparison(run.run_id)}
                        disabled={selectedForComparison.length >= 3 && !selectedForComparison.includes(run.run_id)}
                      />
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" fontFamily="monospace">
                        {run.run_id.substring(0, 8)}...
                      </Typography>
                    </TableCell>
                    <TableCell>
                      {getStatusChip(run)}
                    </TableCell>
                    <TableCell>
                      {formatDate(run.created_at)}
                    </TableCell>
                    <TableCell>
                      {run.total_epochs || '-'}
                    </TableCell>
                    <TableCell>
                      {run.best_val_acc ? `${(run.best_val_acc * 100).toFixed(2)}%` : '-'}
                    </TableCell>
                    <TableCell>
                      {run.final_loss ? run.final_loss.toFixed(4) : '-'}
                    </TableCell>
                    <TableCell>
                      <Box display="flex" gap={1}>
                        <Tooltip title="View Report">
                          <IconButton
                            size="small"
                            onClick={() => handleViewReport(run.run_id)}
                            disabled={!run.has_metrics}
                          >
                            <Assessment />
                          </IconButton>
                        </Tooltip>
                        
                        <Tooltip title="Download Model">
                          <IconButton
                            size="small"
                            onClick={() => handleDownloadModel(run.run_id)}
                            disabled={!run.has_model}
                          >
                            <Download />
                          </IconButton>
                        </Tooltip>
                        
                        <Tooltip title="Delete Run">
                          <IconButton
                            size="small"
                            color="error"
                            onClick={() => {
                              setRunToDelete(run.run_id);
                              setDeleteDialogOpen(true);
                            }}
                          >
                            <Delete />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>

          {runs.length === 0 && !loading && (
            <Box textAlign="center" py={4}>
              <Typography color="textSecondary">
                No training runs found. Start your first training to see results here.
              </Typography>
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Training Report Dialog */}
      {selectedRun && (
        <TrainingReportViewer
          runId={selectedRun}
          open={reportOpen}
          onClose={() => {
            setReportOpen(false);
            setSelectedRun(null);
          }}
        />
      )}

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
        <DialogTitle>Confirm Deletion</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete this training run? This action cannot be undone.
            All associated files including model weights, metrics, and reports will be permanently deleted.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>
            Cancel
          </Button>
          <Button onClick={handleDeleteRun} color="error" variant="contained">
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default TrainingRunsManager;