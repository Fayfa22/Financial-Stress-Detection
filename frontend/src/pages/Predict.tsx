import { useState } from 'react';
import axios from 'axios';
import {
  Container,
  Typography,
  Box,
  Tab,
  TextField,
  Button,
  CircularProgress,
  Alert,
  Card,
  CardContent,
  Slider,
  Grid,
  Paper,
  Divider,
} from '@mui/material';
import { TabContext, TabList, TabPanel } from '@mui/lab';
import { Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
} from 'chart.js';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

ChartJS.register(ArcElement, Tooltip, Legend);

export default function Predict() {
  const [tabValue, setTabValue] = useState('text');

  const [text, setText] = useState('');
  const [numericalFile, setNumericalFile] = useState<File | null>(null);
  const [weightNum, setWeightNum] = useState(60);
  const [weightText, setWeightText] = useState(40);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState('');

  const handleTabChange = (_: React.SyntheticEvent, newValue: string) => {
    setTabValue(newValue);
    setResult(null);
    setError('');
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setNumericalFile(e.target.files[0]);
    }
  };

  const handlePredict = async () => {
    setLoading(true);
    setError('');
    setResult(null);

    try {
      let response;
      const formData = new FormData();

      if (tabValue === 'text') {
        if (!text.trim()) throw new Error("Veuillez entrer du texte pour l'analyse.");
        response = await axios.post('/api/predict/text', { text });
      }
      else if (tabValue === 'numerical') {
        if (!numericalFile) throw new Error('Veuillez sélectionner un fichier contenant les ratios.');
        formData.append('file', numericalFile);
        response = await axios.post('/api/predict/numerical', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
      }
      else if (tabValue === 'fused') {
        if (!text.trim() || !numericalFile) throw new Error('Texte et fichier de ratios requis.');
        formData.append('text', text);
        formData.append('file', numericalFile);
        formData.append('weight_num', (weightNum / 100).toString());
        formData.append('weight_text', (weightText / 100).toString());
        response = await axios.post('/api/predict/fused', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
      }

      setResult(response?.data);
    } catch (err: any) {
      setError(err.message || err.response?.data?.detail || 'Une erreur est survenue lors de la prédiction.');
    } finally {
      setLoading(false);
    }
  };

  const score = result?.fused_score ?? result?.stress_score ?? 0;

  const getChartData = (val: number) => ({
    labels: ['Stress Financier Élevé', 'Sain / Stable'],
    datasets: [{
      data: [val * 100, 100 - val * 100],
      backgroundColor: ['#FF3D71', '#00E676'],
      borderColor: ['rgba(255, 61, 113, 0.2)', 'rgba(0, 230, 118, 0.2)'],
      borderWidth: 2,
      hoverOffset: 4,
    }],
  });

  return (
    <Container maxWidth="md" sx={{ mt: 8, mb: 12 }}>
      <Typography variant="h2" gutterBottom align="center" sx={{ mb: 2 }}>
        Module d'Analyse
      </Typography>
      <Typography variant="body1" color="text.secondary" align="center" sx={{ mb: 6, maxWidth: 600, mx: 'auto' }}>
        Sélectionnez la méthode d'évaluation appropriée. Notre IA traitera vos données de manière sécurisée et instantanée.
      </Typography>

      <Paper sx={{ mb: 6, overflow: 'hidden' }}>
        <TabContext value={tabValue}>
          <Box sx={{ borderBottom: 1, borderColor: 'divider', bgcolor: 'rgba(0,0,0,0.2)' }}>
            <TabList
              onChange={handleTabChange}
              variant="fullWidth"
              sx={{
                '& .MuiTab-root': { py: 3, fontSize: '1.05rem' },
                '& .Mui-selected': { color: 'primary.main', fontWeight: 700 }
              }}
            >
              <Tab label="Analyse Textuelle" value="text" />
              <Tab label="Analyse Numérique" value="numerical" />
              <Tab label="Fusion Multimodale" value="fused" />
            </TabList>
          </Box>

          {/* Onglet Text */}
          <TabPanel value="text" sx={{ p: { xs: 3, md: 5 } }}>
            <Typography variant="h5" gutterBottom>NLP Sémantique</Typography>
            <Typography variant="body2" color="text.secondary" paragraph sx={{ mb: 4 }}>
              Collez le contenu d'un rapport annuel, communiqué de presse ou document stratégique pour détecter les signaux faibles de stress financier.
            </Typography>

            <TextField
              placeholder="Ex: L'entreprise fait face à de graves problèmes de liquidité suite à la baisse des revenus..."
              multiline
              rows={8}
              fullWidth
              value={text}
              onChange={(e) => setText(e.target.value)}
              variant="outlined"
              sx={{ mb: 4 }}
            />

            <Button
              variant="contained"
              fullWidth
              size="large"
              onClick={handlePredict}
              disabled={loading || !text.trim()}
              sx={{ py: 1.5, fontSize: '1.1rem' }}
            >
              {loading ? <CircularProgress size={28} color="inherit" /> : 'Lancer le diagnostic'}
            </Button>
          </TabPanel>

          {/* Onglet Numérique */}
          <TabPanel value="numerical" sx={{ p: { xs: 3, md: 5 } }}>
            <Typography variant="h5" gutterBottom>Modèle Quantitatif</Typography>
            <Typography variant="body2" color="text.secondary" paragraph sx={{ mb: 4 }}>
              Importez un fichier contenant les 64 ratios financiers pour obtenir un score de détresse précis.
            </Typography>

            <Box sx={{ mb: 4, display: 'flex', flexDirection: 'column', alignItems: 'center', p: 4, border: '2px dashed rgba(255,255,255,0.1)', borderRadius: 2, bgcolor: 'rgba(0,0,0,0.2)' }}>
              <input
                accept=".json,.csv,.xlsx,.xls,.arff"
                id="numerical-file"
                type="file"
                style={{ display: 'none' }}
                onChange={handleFileChange}
              />
              <label htmlFor="numerical-file">
                <Button variant="outlined" component="span" startIcon={<CloudUploadIcon />} sx={{ px: 4, py: 1.5 }}>
                  {numericalFile ? numericalFile.name : 'Parcourir les fichiers (.json, .csv, .xlsx, .arff)'}
                </Button>
              </label>
              <Typography variant="caption" color="text.secondary" sx={{ mt: 2 }}>
                Formats acceptés : JSON, CSV, Excel, ARFF (max 5 Mo)
              </Typography>
            </Box>

            <Button
              variant="contained"
              fullWidth
              size="large"
              onClick={handlePredict}
              disabled={loading || !numericalFile}
              sx={{ py: 1.5, fontSize: '1.1rem' }}
            >
              {loading ? <CircularProgress size={28} color="inherit" /> : 'Lancer le diagnostic'}
            </Button>
          </TabPanel>

          {/* Onglet Fusionné */}
          <TabPanel value="fused" sx={{ p: { xs: 3, md: 5 } }}>
            <Typography variant="h5" gutterBottom>Intelligence Combinée</Typography>
            <Typography variant="body2" color="text.secondary" paragraph sx={{ mb: 4 }}>
              Pousse l'analyse à son maximum en croisant les signaux NLP et quantitatifs. Ajustez la pondération selon votre cas d'usage.
            </Typography>

            <Grid container spacing={4} sx={{ mb: 4 }}>
              <Grid size={{ xs: 12, md: 6 }}>
                <Typography variant="subtitle2" gutterBottom color="text.secondary">1. Données Sémantiques</Typography>
                <TextField
                  placeholder="Texte financier..."
                  multiline
                  rows={6}
                  fullWidth
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  variant="outlined"
                />
              </Grid>

              <Grid size={{ xs: 12, md: 6 }}>
                <Typography variant="subtitle2" gutterBottom color="text.secondary">2. Données Quantitatives</Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '165px', border: '1px dashed rgba(255,255,255,0.2)', borderRadius: 2, bgcolor: 'rgba(0,0,0,0.1)' }}>
                  <input
                    accept=".json,.csv,.xlsx,.xls,.arff"
                    id="fused-file"
                    type="file"
                    style={{ display: 'none' }}
                    onChange={handleFileChange}
                  />
                  <label htmlFor="fused-file">
                    <Button variant="outlined" component="span" startIcon={<CloudUploadIcon />}>
                      {numericalFile ? numericalFile.name : 'Choisir fichier (.json, .csv, .xlsx, .arff)'}
                    </Button>
                  </label>
                </Box>
              </Grid>
            </Grid>

            <Divider sx={{ my: 4 }} />

            <Typography variant="subtitle1" gutterBottom fontWeight={600}>
              Paramètres de Pondération
            </Typography>
            <Grid container spacing={5} sx={{ mb: 5 }}>
              <Grid size={{ xs: 12, sm: 6 }}>
                <Typography id="weight-num" color="text.secondary" gutterBottom>Importance Quantitative ({weightNum}%)</Typography>
                <Slider
                  value={weightNum}
                  onChange={(_, v) => {
                    setWeightNum(v as number);
                    setWeightText(100 - (v as number));
                  }}
                  aria-labelledby="weight-num"
                  min={0}
                  max={100}
                  step={5}
                  color="secondary"
                />
              </Grid>
              <Grid size={{ xs: 12, sm: 6 }}>
                <Typography id="weight-text" color="text.secondary" gutterBottom>Importance Sémantique ({weightText}%)</Typography>
                <Slider
                  value={weightText}
                  onChange={(_, v) => {
                    setWeightText(v as number);
                    setWeightNum(100 - (v as number));
                  }}
                  aria-labelledby="weight-text"
                  min={0}
                  max={100}
                  step={5}
                  color="primary"
                />
              </Grid>
            </Grid>

            <Button
              variant="contained"
              fullWidth
              size="large"
              onClick={handlePredict}
              disabled={loading || !text.trim() || !numericalFile}
              sx={{ py: 1.5, fontSize: '1.1rem' }}
            >
              {loading ? <CircularProgress size={28} color="inherit" /> : 'Exécuter la fusion multimodale'}
            </Button>
          </TabPanel>
        </TabContext>
      </Paper>

      {error && (
        <Alert severity="error" variant="filled" sx={{ mt: 4, borderRadius: 2 }}>
          {error}
        </Alert>
      )}

      {result && (
        <Card sx={{ mt: 6, overflow: 'visible', position: 'relative' }}>
          <Box sx={{
            position: 'absolute',
            top: -20,
            left: '50%',
            transform: 'translateX(-50%)',
            background: score > 0.5 ? 'linear-gradient(135deg, #FF3D71 0%, #B71C1C 100%)' : 'linear-gradient(135deg, #00E676 0%, #00B248 100%)',
            color: '#fff',
            px: 4,
            py: 1,
            borderRadius: 8,
            boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
            fontWeight: 800,
            letterSpacing: '1px',
            textTransform: 'uppercase'
          }}>
            {score > 0.5 ? 'Alerte Stress Financier' : 'Situation Stable'}
          </Box>
          <CardContent sx={{ pt: 6, pb: 4, px: { xs: 3, md: 6 } }}>
            <Grid container spacing={4} alignItems="center">
              <Grid size={{ xs: 12, md: 6 }}>
                <Box sx={{ position: 'relative', width: '100%', maxWidth: '300px', mx: 'auto' }}>
                  <Doughnut
                    data={getChartData(score)}
                    options={{
                      responsive: true,
                      cutout: '75%',
                      plugins: {
                        legend: { position: 'bottom', labels: { color: '#F3F4F6', padding: 20 } },
                        tooltip: { backgroundColor: 'rgba(0,0,0,0.8)', titleFont: { size: 14 }, bodyFont: { size: 14 } },
                      },
                    }}
                  />
                  <Box sx={{
                    position: 'absolute',
                    top: '45%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    textAlign: 'center'
                  }}>
                    <Typography variant="h3" fontWeight={800} color={score > 0.5 ? '#FF3D71' : '#00E676'}>
                      {(score * 100).toFixed(1)}%
                    </Typography>
                    <Typography variant="caption" color="text.secondary">Indice de Risque</Typography>
                  </Box>
                </Box>
              </Grid>

              <Grid size={{ xs: 12, md: 6 }}>
                <Typography variant="h5" gutterBottom sx={{ mb: 3 }}>Détails de l'analyse</Typography>

                <Box sx={{ mb: 2, p: 2, borderRadius: 2, bgcolor: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.05)' }}>
                  <Typography variant="body2" color="text.secondary">Niveau détecté</Typography>
                  <Typography variant="h6" color={score > 0.5 ? '#FF3D71' : '#00E676'}>
                    {result.interpretation?.level || (score > 0.5 ? 'Critique' : 'Sain')}
                  </Typography>
                </Box>

                <Box sx={{ mb: 2, p: 2, borderRadius: 2, bgcolor: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.05)' }}>
                  <Typography variant="body2" color="text.secondary">Message</Typography>
                  <Typography variant="body1">
                    {result.interpretation?.message || '—'}
                  </Typography>
                </Box>

                {tabValue === 'fused' && (
                  <>
                    <Box sx={{ mb: 2, p: 2, borderRadius: 2, bgcolor: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.05)' }}>
                      <Typography variant="body2" color="text.secondary">Score Quantitatif</Typography>
                      <Typography variant="h6">
                        {result.numerical_score !== undefined ? (result.numerical_score * 100).toFixed(1) + '%' : 'N/A'}
                      </Typography>
                    </Box>
                    <Box sx={{ mb: 2, p: 2, borderRadius: 2, bgcolor: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.05)' }}>
                      <Typography variant="body2" color="text.secondary">Score Textuel (NLP)</Typography>
                      <Typography variant="h6">
                        {result.textual_score !== undefined ? (result.textual_score * 100).toFixed(1) + '%' : 'N/A'}
                      </Typography>
                    </Box>
                    {result.alert && (
                      <Alert severity="warning" sx={{ mt: 1 }}>
                        {result.alert}
                      </Alert>
                    )}
                  </>
                )}

                {tabValue === 'text' && result.probabilities && (
                  <Box sx={{ p: 2, borderRadius: 2, bgcolor: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.05)' }}>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>Probabilités</Typography>
                    {Object.entries(result.probabilities).map(([label, prob]: [string, any]) => (
                      <Typography variant="body2" key={label}>
                        {label} : {(prob * 100).toFixed(1)}%
                      </Typography>
                    ))}
                  </Box>
                )}
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      )}
    </Container>
  );
}