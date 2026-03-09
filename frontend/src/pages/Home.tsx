import { Container, Typography, Box, Button, Grid, Paper } from '@mui/material';
import { Link } from 'react-router-dom';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import AutoGraphIcon from '@mui/icons-material/AutoGraph';
import SecurityIcon from '@mui/icons-material/Security';

export default function Home() {
  const features = [
    {
      title: 'Analyse Textuelle NLP',
      desc: 'Détection du stress via l\'analyse sémantique des rapports annuels et communiqués financiers.',
      icon: <AnalyticsIcon sx={{ fontSize: 40, color: 'primary.main' }} />
    },
    {
      title: 'Modélisation Numérique',
      desc: 'Évaluation de plus de 60 ratios financiers pour diagnostiquer la rentabilité et la solvabilité.',
      icon: <AutoGraphIcon sx={{ fontSize: 40, color: 'secondary.main' }} />
    },
    {
      title: 'Fusion Multimodale',
      desc: 'Une précision inégalée en combinant les signaux textuels et quantitatifs en temps réel.',
      icon: <SecurityIcon sx={{ fontSize: 40, color: '#D4AF37' }} /> // Gold touch
    }
  ];

  return (
    <Box sx={{ overflow: 'hidden' }}>
      <Container maxWidth="lg" sx={{ pt: { xs: 8, md: 15 }, pb: { xs: 8, md: 12 } }}>
        <Grid container spacing={6} alignItems="center">
          <Grid size={{ xs: 12, md: 7 }}>
            <Typography variant="h1" sx={{ fontSize: { xs: '3rem', md: '4.5rem' }, fontWeight: 800, mb: 3, lineHeight: 1.1 }}>
              Anticipez le <br />
              <span style={{ color: '#00E676' }}>Stress Financier</span>
            </Typography>
            <Typography variant="h5" color="text.secondary" sx={{ mb: 5, fontWeight: 400, maxWidth: '600px', lineHeight: 1.6 }}>
              La plateforme d'intelligence artificielle de nouvelle génération pour détecter la vulnérabilité des entreprises avant qu'il ne soit trop tard.
            </Typography>
            <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
              <Button 
                variant="contained" 
                size="large" 
                component={Link} 
                to="/predict"
                sx={{ py: 1.5, px: 4, fontSize: '1.1rem' }}
              >
                Démarrer l'Analyse
              </Button>
              <Button 
                variant="outlined" 
                size="large" 
                sx={{ py: 1.5, px: 4, fontSize: '1.1rem', color: '#F3F4F6' }}
              >
                En savoir plus
              </Button>
            </Box>
          </Grid>
          
          <Grid size={{ xs: 12, md: 5 }} sx={{ display: { xs: 'none', md: 'block' } }}>
            {/* Abstract decorative element representing data/finance */}
            <Box sx={{ position: 'relative', height: '400px', width: '100%' }}>
              <Box sx={{
                position: 'absolute',
                top: 0, right: 0, bottom: 0, left: 0,
                background: 'linear-gradient(45deg, rgba(0,230,118,0.2) 0%, rgba(41,121,255,0.2) 100%)',
                borderRadius: '30% 70% 70% 30% / 30% 30% 70% 70%',
                filter: 'blur(40px)',
                animation: 'float 6s ease-in-out infinite'
              }} />
              <style>
                {`@keyframes float { 0% { transform: translateY(0px) rotate(0deg); } 50% { transform: translateY(-20px) rotate(5deg); } 100% { transform: translateY(0px) rotate(0deg); } }`}
              </style>
            </Box>
          </Grid>
        </Grid>

        <Box sx={{ mt: { xs: 10, md: 15 } }}>
          <Typography variant="h3" align="center" gutterBottom sx={{ mb: 8, fontWeight: 700 }}>
            Une approche innovante
          </Typography>
          <Grid container spacing={4}>
            {features.map((feat, idx) => (
              <Grid size={{ xs: 12, md: 4 }} key={idx}>
                <Paper sx={{ p: 5, height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center', transition: 'transform 0.3s', '&:hover': { transform: 'translateY(-10px)' } }}>
                  <Box sx={{ mb: 3, p: 2, borderRadius: '50%', background: 'rgba(255,255,255,0.03)' }}>
                    {feat.icon}
                  </Box>
                  <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                    {feat.title}
                  </Typography>
                  <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.7 }}>
                    {feat.desc}
                  </Typography>
                </Paper>
              </Grid>
            ))}
          </Grid>
        </Box>
      </Container>
    </Box>
  );
}