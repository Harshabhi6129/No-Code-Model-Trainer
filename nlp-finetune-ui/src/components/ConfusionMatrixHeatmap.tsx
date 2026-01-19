// src/components/ConfusionMatrixHeatmap.tsx
import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Tooltip
} from '@mui/material';

interface ConfusionMatrixProps {
  matrix: number[][];
  labels?: string[];
  title?: string;
}

const ConfusionMatrixHeatmap: React.FC<ConfusionMatrixProps> = ({
  matrix,
  labels,
  title = "Confusion Matrix"
}) => {
  if (!matrix || matrix.length === 0) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            {title}
          </Typography>
          <Typography color="textSecondary">
            No confusion matrix data available
          </Typography>
        </CardContent>
      </Card>
    );
  }

  // Calculate total predictions for percentages
  const total = matrix.flat().reduce((sum, val) => sum + val, 0);
  
  // Find max value for color scaling
  const maxValue = Math.max(...matrix.flat());
  
  // Generate labels if not provided
  const classLabels = labels || matrix.map((_, i) => `Class ${i}`);
  
  // Calculate accuracy metrics
  const correctPredictions = matrix.reduce((sum, row, i) => sum + (row[i] || 0), 0);
  const accuracy = total > 0 ? (correctPredictions / total) * 100 : 0;
  
  // Calculate per-class metrics
  const classMetrics = classLabels.map((label, i) => {
    const truePositives = matrix[i]?.[i] || 0;
    const falsePositives = matrix.reduce((sum, row, j) => j !== i ? sum + (row[i] || 0) : sum, 0);
    const falseNegatives = matrix[i]?.reduce((sum, val, j) => j !== i ? sum + val : sum, 0) || 0;
    
    const precision = truePositives + falsePositives > 0 ? truePositives / (truePositives + falsePositives) : 0;
    const recall = truePositives + falseNegatives > 0 ? truePositives / (truePositives + falseNegatives) : 0;
    const f1 = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
    
    return { label, precision, recall, f1, truePositives };
  });

  const getCellColor = (value: number) => {
    const intensity = maxValue > 0 ? value / maxValue : 0;
    const opacity = 0.1 + (intensity * 0.8); // Scale from 0.1 to 0.9
    return `rgba(25, 118, 210, ${opacity})`; // Material-UI primary blue
  };

  const getCellTextColor = (value: number) => {
    const intensity = maxValue > 0 ? value / maxValue : 0;
    return intensity > 0.5 ? 'white' : 'black';
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          {title}
        </Typography>
        
        {/* Overall Accuracy */}
        <Box mb={2}>
          <Typography variant="body2" color="textSecondary">
            Overall Accuracy: <strong>{accuracy.toFixed(2)}%</strong> ({correctPredictions}/{total})
          </Typography>
        </Box>

        {/* Confusion Matrix */}
        <TableContainer component={Paper} variant="outlined" sx={{ mb: 2 }}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell sx={{ fontWeight: 'bold', bgcolor: 'grey.100' }}>
                  Predicted →<br />Actual ↓
                </TableCell>
                {classLabels.map((label, i) => (
                  <TableCell 
                    key={i} 
                    align="center" 
                    sx={{ fontWeight: 'bold', bgcolor: 'grey.100' }}
                  >
                    {label}
                  </TableCell>
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              {matrix.map((row, i) => (
                <TableRow key={i}>
                  <TableCell 
                    component="th" 
                    scope="row" 
                    sx={{ fontWeight: 'bold', bgcolor: 'grey.100' }}
                  >
                    {classLabels[i]}
                  </TableCell>
                  {row.map((value, j) => {
                    const percentage = total > 0 ? (value / total * 100).toFixed(1) : '0.0';
                    const isCorrect = i === j;
                    
                    return (
                      <Tooltip
                        key={j}
                        title={`${classLabels[i]} → ${classLabels[j]}: ${value} (${percentage}%)`}
                      >
                        <TableCell
                          align="center"
                          sx={{
                            bgcolor: getCellColor(value),
                            color: getCellTextColor(value),
                            fontWeight: isCorrect ? 'bold' : 'normal',
                            border: isCorrect ? '2px solid #1976d2' : '1px solid #e0e0e0',
                            cursor: 'pointer',
                            '&:hover': {
                              bgcolor: getCellColor(value * 1.2),
                            }
                          }}
                        >
                          <Box>
                            <Typography variant="body2" component="div">
                              {value}
                            </Typography>
                            <Typography variant="caption" component="div">
                              {percentage}%
                            </Typography>
                          </Box>
                        </TableCell>
                      </Tooltip>
                    );
                  })}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>

        {/* Per-Class Metrics */}
        <Typography variant="subtitle2" gutterBottom>
          Per-Class Metrics
        </Typography>
        <TableContainer component={Paper} variant="outlined">
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell sx={{ fontWeight: 'bold' }}>Class</TableCell>
                <TableCell align="right" sx={{ fontWeight: 'bold' }}>Precision</TableCell>
                <TableCell align="right" sx={{ fontWeight: 'bold' }}>Recall</TableCell>
                <TableCell align="right" sx={{ fontWeight: 'bold' }}>F1-Score</TableCell>
                <TableCell align="right" sx={{ fontWeight: 'bold' }}>Support</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {classMetrics.map((metrics, i) => (
                <TableRow key={i}>
                  <TableCell component="th" scope="row">
                    {metrics.label}
                  </TableCell>
                  <TableCell align="right">
                    {(metrics.precision * 100).toFixed(1)}%
                  </TableCell>
                  <TableCell align="right">
                    {(metrics.recall * 100).toFixed(1)}%
                  </TableCell>
                  <TableCell align="right">
                    {(metrics.f1 * 100).toFixed(1)}%
                  </TableCell>
                  <TableCell align="right">
                    {matrix[i]?.reduce((sum, val) => sum + val, 0) || 0}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </CardContent>
    </Card>
  );
};

export default ConfusionMatrixHeatmap;