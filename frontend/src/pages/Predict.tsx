import { useState } from 'react';
import axios from 'axios';
import {
  Container, Typography, Box, Tab, TextField, Button,
  CircularProgress, Alert, Card, CardContent, Slider,
  Grid, Paper, Divider, Chip,
} from '@mui/material';
import { TabContext, TabList, TabPanel } from '@mui/lab';
import { Doughnut, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS, ArcElement, Tooltip, Legend,
  CategoryScale, LinearScale, BarElement,
} from 'chart.js';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import PsychologyIcon from '@mui/icons-material/Psychology';

ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement);

export default function Predict() {
  const [tabValue, setTabValue] = useState('text');
  const [text, setText] = useState('');
  const [numericalFile, setNumericalFile] = useState<File | null>(null);
  const [weightNum, setWeightNum] = useState(60);
  const [weightText, setWeightText] = useState(40);
  const [loading, setLoading] = useState(false);
  const [loadingExplain, setLoadingExplain] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [explainResult, setExplainResult] = useState<any>(null);
  const [error, setError] = useState('');

  const handleTabChange = (_: React.SyntheticEvent, newValue: string) => {
    setTabValue(newValue);
    setResult(null);
    setExplainResult(null);
    setError('');
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setNumericalFile(e.target.files[0]);
      setExplainResult(null);
    }
  };

  const handlePredict = async () => {
    setLoading(true);
    setError('');
    setResult(null);
    setExplainResult(null);
    try {
      let response;
      const formData = new FormData();
      if (tabValue === 'text') {
        if (!text.trim()) throw new Error("Veuillez entrer du texte pour l'analyse.");
        response = await axios.post('/api/predict/text', { text });
      } else if (tabValue === 'numerical') {
        if (!numericalFile) throw new Error('Veuillez sélectionner un fichier.');
        formData.append('file', numericalFile);
        response = await axios.post('/api/predict/numerical', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
      } else if (tabValue === 'fused') {
        if (!text.trim() || !numericalFile) throw new Error('Texte et fichier requis.');
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
      setError(err.message || err.response?.data?.detail || 'Une erreur est survenue.');
    } finally {
      setLoading(false);
    }
  };

  const handleExplain = async () => {
    setLoadingExplain(true);
    setExplainResult(null);
    try {
      if (tabValue === 'numerical' && numericalFile) {
        const formData = new FormData();
        formData.append('file', numericalFile);
        const res = await axios.post('/api/explain/numerical', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
        setExplainResult({ type: 'numerical', data: res.data });
      } else if (tabValue === 'text' && text.trim()) {
        const res = await axios.post('/api/explain/text', { text });
        setExplainResult({ type: 'text', data: res.data });
      } else if (tabValue === 'fused') {
        const formData = new FormData();
        formData.append('file', numericalFile!);
        formData.append('text', text);
        const res = await axios.post('/api/explain/fused', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
        setExplainResult({
          type: 'fused',
          dataNum: res.data.numerical,
          dataTxt: res.data.textual,
        });
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Erreur explicabilité.');
    } finally {
      setLoadingExplain(false);
    }
  };

  const score = result?.fused_score ?? result?.stress_score ?? 0;

  const getChartData = (val: number) => ({
    labels: ['Stress Financier', 'Sain / Stable'],
    datasets: [{
      data: [val * 100, 100 - val * 100],
      backgroundColor: ['#FF3D71', '#00E676'],
      borderColor: ['rgba(255,61,113,0.2)', 'rgba(0,230,118,0.2)'],
      borderWidth: 2,
      hoverOffset: 4,
    }],
  });

  const getShapChartData = (shapValues: any[]) => ({
    labels: shapValues.map(v => v.feature),
    datasets: [{
      label: 'Impact SHAP',
      data: shapValues.map(v => v.impact),
      backgroundColor: shapValues.map(v => v.impact >= 0 ? 'rgba(255,61,113,0.7)' : 'rgba(0,230,118,0.7)'),
      borderColor: shapValues.map(v => v.impact >= 0 ? '#FF3D71' : '#00E676'),
      borderWidth: 1,
    }],
  });

  const statBox = (label: string, value: string, color?: string) => (
    <Box sx={{ mb: 2, p: 2, borderRadius: 2, bgcolor: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.05)' }}>
      <Typography variant="body2" color="text.secondary">{label}</Typography>
      <Typography variant="h6" color={color}>{value}</Typography>
    </Box>
  );

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
            <TabList onChange={handleTabChange} variant="fullWidth" sx={{
              '& .MuiTab-root': { py: 3, fontSize: '1.05rem' },
              '& .Mui-selected': { color: 'primary.main', fontWeight: 700 }
            }}>
              <Tab label="Analyse Textuelle" value="text" />
              <Tab label="Analyse Numérique" value="numerical" />
              <Tab label="Fusion Multimodale" value="fused" />
            </TabList>
          </Box>

          {/* ── Onglet Text ── */}
          <TabPanel value="text" sx={{ p: { xs: 3, md: 5 } }}>
            <Typography variant="h5" gutterBottom>NLP Sémantique</Typography>
            <Typography variant="body2" color="text.secondary" paragraph sx={{ mb: 4 }}>
              Collez le contenu d'un rapport annuel, communiqué de presse ou document stratégique.
            </Typography>
            <TextField
              placeholder="Ex: The company faces serious liquidity issues due to declining revenues..."
              multiline rows={8} fullWidth value={text}
              onChange={(e) => setText(e.target.value)}
              variant="outlined" sx={{ mb: 4 }}
            />
            <Button variant="contained" fullWidth size="large" onClick={handlePredict}
              disabled={loading || !text.trim()} sx={{ py: 1.5, fontSize: '1.1rem' }}>
              {loading ? <CircularProgress size={28} color="inherit" /> : 'Lancer le diagnostic'}
            </Button>
          </TabPanel>

          {/* ── Onglet Numérique ── */}
          <TabPanel value="numerical" sx={{ p: { xs: 3, md: 5 } }}>
            <Typography variant="h5" gutterBottom>Modèle Quantitatif</Typography>
            <Typography variant="body2" color="text.secondary" paragraph sx={{ mb: 4 }}>
              Importez un fichier contenant les ratios financiers pour obtenir un score de détresse précis.
            </Typography>
            <Box sx={{ mb: 4, display: 'flex', flexDirection: 'column', alignItems: 'center', p: 4, border: '2px dashed rgba(255,255,255,0.1)', borderRadius: 2, bgcolor: 'rgba(0,0,0,0.2)' }}>
              <input accept=".json,.csv,.xlsx,.xls,.arff" id="numerical-file" type="file"
                style={{ display: 'none' }} onChange={handleFileChange} />
              <label htmlFor="numerical-file">
                <Button variant="outlined" component="span" startIcon={<CloudUploadIcon />} sx={{ px: 4, py: 1.5 }}>
                  {numericalFile ? numericalFile.name : 'Parcourir les fichiers (.json, .csv, .xlsx, .arff)'}
                </Button>
              </label>
              <Typography variant="caption" color="text.secondary" sx={{ mt: 2 }}>
                Formats acceptés : JSON, CSV, Excel, ARFF (max 5 Mo)
              </Typography>
            </Box>
            <Button variant="contained" fullWidth size="large" onClick={handlePredict}
              disabled={loading || !numericalFile} sx={{ py: 1.5, fontSize: '1.1rem' }}>
              {loading ? <CircularProgress size={28} color="inherit" /> : 'Lancer le diagnostic'}
            </Button>
          </TabPanel>

          {/* ── Onglet Fusionné ── */}
          <TabPanel value="fused" sx={{ p: { xs: 3, md: 5 } }}>
            <Typography variant="h5" gutterBottom>Intelligence Combinée</Typography>
            <Typography variant="body2" color="text.secondary" paragraph sx={{ mb: 4 }}>
              Croisez les signaux NLP et quantitatifs. Ajustez la pondération selon votre cas d'usage.
            </Typography>
            <Grid container spacing={4} sx={{ mb: 4 }}>
              <Grid size={{ xs: 12, md: 6 }}>
                <Typography variant="subtitle2" gutterBottom color="text.secondary">1. Données Sémantiques</Typography>
                <TextField placeholder="Texte financier..." multiline rows={6} fullWidth
                  value={text} onChange={(e) => setText(e.target.value)} variant="outlined" />
              </Grid>
              <Grid size={{ xs: 12, md: 6 }}>
                <Typography variant="subtitle2" gutterBottom color="text.secondary">2. Données Quantitatives</Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '165px', border: '1px dashed rgba(255,255,255,0.2)', borderRadius: 2, bgcolor: 'rgba(0,0,0,0.1)' }}>
                  <input accept=".json,.csv,.xlsx,.xls,.arff" id="fused-file" type="file"
                    style={{ display: 'none' }} onChange={handleFileChange} />
                  <label htmlFor="fused-file">
                    <Button variant="outlined" component="span" startIcon={<CloudUploadIcon />}>
                      {numericalFile ? numericalFile.name : 'Choisir fichier (.json, .csv, .xlsx, .arff)'}
                    </Button>
                  </label>
                </Box>
              </Grid>
            </Grid>
            <Divider sx={{ my: 4 }} />
            <Typography variant="subtitle1" gutterBottom fontWeight={600}>Paramètres de Pondération</Typography>
            <Grid container spacing={5} sx={{ mb: 5 }}>
              <Grid size={{ xs: 12, sm: 6 }}>
                <Typography color="text.secondary" gutterBottom>Importance Quantitative ({weightNum}%)</Typography>
                <Slider value={weightNum} onChange={(_, v) => { setWeightNum(v as number); setWeightText(100 - (v as number)); }}
                  min={0} max={100} step={5} color="secondary" />
              </Grid>
              <Grid size={{ xs: 12, sm: 6 }}>
                <Typography color="text.secondary" gutterBottom>Importance Sémantique ({weightText}%)</Typography>
                <Slider value={weightText} onChange={(_, v) => { setWeightText(v as number); setWeightNum(100 - (v as number)); }}
                  min={0} max={100} step={5} color="primary" />
              </Grid>
            </Grid>
            <Button variant="contained" fullWidth size="large" onClick={handlePredict}
              disabled={loading || !text.trim() || !numericalFile} sx={{ py: 1.5, fontSize: '1.1rem' }}>
              {loading ? <CircularProgress size={28} color="inherit" /> : 'Exécuter la fusion multimodale'}
            </Button>
          </TabPanel>
        </TabContext>
      </Paper>

      {error && <Alert severity="error" variant="filled" sx={{ mt: 4, borderRadius: 2 }}>{error}</Alert>}

      {/* ══════════ CARTE RÉSULTAT ══════════ */}
      {result && (
        <Card sx={{ mt: 6, overflow: 'visible', position: 'relative' }}>
          <Box sx={{
            position: 'absolute', top: -20, left: '50%', transform: 'translateX(-50%)',
            background: score > 0.5 ? 'linear-gradient(135deg, #FF3D71 0%, #B71C1C 100%)' : 'linear-gradient(135deg, #00E676 0%, #00B248 100%)',
            color: '#fff', px: 4, py: 1, borderRadius: 8,
            boxShadow: '0 4px 20px rgba(0,0,0,0.5)', fontWeight: 800, letterSpacing: '1px', textTransform: 'uppercase'
          }}>
            {result.interpretation?.emoji} {score > 0.5 ? 'Alerte Stress Financier' : 'Situation Stable'}
          </Box>

          <CardContent sx={{ pt: 6, pb: 4, px: { xs: 3, md: 6 } }}>
            <Grid container spacing={4} alignItems="center">
              {/* Donut */}
              <Grid size={{ xs: 12, md: 6 }}>
                <Box sx={{ position: 'relative', width: '100%', maxWidth: '300px', mx: 'auto' }}>
                  <Doughnut data={getChartData(score)} options={{
                    responsive: true, cutout: '75%',
                    plugins: {
                      legend: { position: 'bottom', labels: { color: '#F3F4F6', padding: 20 } },
                      tooltip: { backgroundColor: 'rgba(0,0,0,0.8)' },
                    },
                  }} />
                  <Box sx={{ position: 'absolute', top: '45%', left: '50%', transform: 'translate(-50%, -50%)', textAlign: 'center' }}>
                    <Typography variant="h3" fontWeight={800} color={score > 0.5 ? '#FF3D71' : '#00E676'}>
                      {(score * 100).toFixed(1)}%
                    </Typography>
                    <Typography variant="caption" color="text.secondary">Indice de Risque</Typography>
                  </Box>
                </Box>
              </Grid>

              {/* Détails */}
              <Grid size={{ xs: 12, md: 6 }}>
                <Typography variant="h5" gutterBottom sx={{ mb: 3 }}>Détails de l'analyse</Typography>

                {statBox('Niveau détecté', `${result.interpretation?.emoji || ''} ${result.interpretation?.level || '—'}`, score > 0.5 ? '#FF3D71' : '#00E676')}
                {statBox('Message', result.interpretation?.message || '—')}

                {/* Spécifique TEXTUEL */}
                {tabValue === 'text' && (
                  <>
                    {statBox('Sentiment', result.sentiment || '—')}
                    {statBox('Confiance', result.confidence !== undefined ? (result.confidence * 100).toFixed(1) + '%' : '—')}
                    {result.probabilities && (
                      <Box sx={{ mb: 2, p: 2, borderRadius: 2, bgcolor: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.05)' }}>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>Probabilités par classe</Typography>
                        {Object.entries(result.probabilities).map(([label, prob]: [string, any]) => (
                          <Box key={label} sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                            <Typography variant="body2">{label}</Typography>
                            <Typography variant="body2" fontWeight={700}>{(prob * 100).toFixed(1)}%</Typography>
                          </Box>
                        ))}
                      </Box>
                    )}
                  </>
                )}

                {/* Spécifique NUMÉRIQUE */}
                {tabValue === 'numerical' && (
                  <>
                    {statBox('Prédiction', result.prediction || '—')}
                    {statBox('Confiance', result.confidence !== undefined ? (result.confidence * 100).toFixed(1) + '%' : '—')}
                  </>
                )}

                {/* Spécifique FUSIONNÉ */}
                {tabValue === 'fused' && (
                  <>
                    {statBox('Score Quantitatif', result.numerical_score !== undefined ? (result.numerical_score * 100).toFixed(1) + '%' : 'N/A')}
                    {statBox('Score Textuel (NLP)', result.textual_score !== undefined ? (result.textual_score * 100).toFixed(1) + '%' : 'N/A')}
                    <Box sx={{ mb: 2, p: 2, borderRadius: 2, bgcolor: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.05)' }}>
                      <Typography variant="body2" color="text.secondary">Pondération appliquée</Typography>
                      <Typography variant="body1">
                        Numérique {result.weight_num ? (result.weight_num * 100).toFixed(0) : weightNum}% / Textuel {result.weight_text ? (result.weight_text * 100).toFixed(0) : weightText}%
                      </Typography>
                    </Box>
                    <Box sx={{ mb: 2, p: 2, borderRadius: 2, bgcolor: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.05)' }}>
                      <Typography variant="body2" color="text.secondary">Divergence entre signaux</Typography>
                      <Typography variant="h6" color={result.divergence > 0.4 ? '#FF9800' : '#00E676'}>
                        {result.divergence !== undefined ? (result.divergence * 100).toFixed(1) + '%' : '—'}
                      </Typography>
                    </Box>
                    {result.alert && <Alert severity="warning" sx={{ mt: 1 }}>{result.alert}</Alert>}
                  </>
                )}
              </Grid>
            </Grid>

            {/* ══════════ BOUTON EXPLIQUER ══════════ */}
            <Divider sx={{ my: 4 }} />
            <Box sx={{ textAlign: 'center' }}>
              <Button
                variant="outlined"
                size="large"
                startIcon={loadingExplain ? <CircularProgress size={18} /> : <PsychologyIcon />}
                onClick={handleExplain}
                disabled={loadingExplain}
                sx={{ px: 6, py: 1.5, borderColor: 'primary.main', color: 'primary.main', '&:hover': { bgcolor: 'rgba(99,102,241,0.08)' } }}
              >
                {loadingExplain ? 'Analyse en cours...' : '🔍 Expliquer la décision (XAI)'}
              </Button>
            </Box>

            {/* ══════════ SECTION XAI ══════════ */}
            {explainResult && (
              <Box sx={{ mt: 4 }}>
                <Divider sx={{ mb: 4 }} />
                <Typography variant="h5" gutterBottom sx={{ mb: 3 }}>
                  🧠 Explicabilité de la décision
                </Typography>

                {/* SHAP numérique */}
                {(explainResult.type === 'numerical' || explainResult.type === 'fused') && explainResult.dataNum?.shap_values && (
                  <Box sx={{ mb: 4 }}>
                    <Typography variant="h6" gutterBottom color="primary.main">
                      📊 Impact des ratios financiers (SHAP)
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      Les barres rouges augmentent le risque de stress, les vertes le diminuent.
                    </Typography>
                    <Box sx={{ bgcolor: 'rgba(0,0,0,0.2)', borderRadius: 2, p: 2 }}>
                      <Bar
                        data={getShapChartData(explainResult.dataNum.shap_values)}
                        options={{
                          indexAxis: 'y',
                          responsive: true,
                          plugins: {
                            legend: { display: false },
                            tooltip: {
                              callbacks: {
                                label: (ctx) => `Impact: ${(ctx.parsed.x ?? 0) > 0 ? '+' : ''}${(ctx.parsed.x ?? 0).toFixed(4)}`
                              }
                            }
                          },
                          scales: {
                            x: { ticks: { color: '#9CA3AF' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                            y: { ticks: { color: '#F3F4F6', font: { size: 11 } }, grid: { display: false } },
                          },
                        }}
                      />
                    </Box>
                    <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                      Valeur de base (score moyen du modèle) : {explainResult.dataNum.base_value?.toFixed(3)}
                    </Typography>
                  </Box>
                )}

                {/* Mots importants textuel */}
                {(explainResult.type === 'text' || explainResult.type === 'fused') && explainResult.dataTxt?.top_words && (
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="h6" gutterBottom color="primary.main">
                      📝 Mots clés détectés (TF-IDF)
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      Les mots avec le plus d'influence sur la prédiction textuelle.
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                      {explainResult.dataTxt.top_words.map((item: any, idx: number) => (
                        <Chip
                          key={idx}
                          label={`${item.word} (${item.score.toFixed(3)})`}
                          sx={{
                            bgcolor: `rgba(99,102,241,${0.3 + idx * 0.05})`,
                            color: '#fff',
                            fontWeight: idx < 3 ? 700 : 400,
                            fontSize: idx < 3 ? '0.9rem' : '0.8rem',
                          }}
                        />
                      ))}
                    </Box>
                    {explainResult.dataTxt.cleaned_text && (
                      <Box sx={{ p: 2, borderRadius: 2, bgcolor: 'rgba(0,0,0,0.2)', border: '1px solid rgba(255,255,255,0.05)' }}>
                        <Typography variant="caption" color="text.secondary">Texte après prétraitement :</Typography>
                        <Typography variant="body2" sx={{ mt: 0.5, fontStyle: 'italic', color: '#9CA3AF' }}>
                          {explainResult.dataTxt.cleaned_text}
                        </Typography>
                      </Box>
                    )}
                  </Box>
                )}
              </Box>
            )}
          </CardContent>
        </Card>
      )}
    </Container>
  );
}