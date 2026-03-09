import { AppBar, Toolbar, Typography, Button, Box, Container } from '@mui/material';
import { Link, useLocation } from 'react-router-dom';
import ShowChartIcon from '@mui/icons-material/ShowChart';

export default function Navbar() {
  const location = useLocation();

  const navItemStyle = (path: string) => ({
    mx: 1,
    color: location.pathname === path ? 'primary.main' : 'text.primary',
    fontWeight: location.pathname === path ? 700 : 500,
    textTransform: 'none',
    fontSize: '1rem',
    '&:hover': {
      color: 'primary.light',
      background: 'rgba(0, 230, 118, 0.08)',
    },
  });

  return (
    <AppBar position="sticky" elevation={0} sx={{ top: 0, zIndex: 1200 }}>
      <Container maxWidth="lg">
        <Toolbar disableGutters sx={{ minHeight: '80px !important' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', flexGrow: 1, textDecoration: 'none', color: 'inherit' }} component={Link} to="/">
            <ShowChartIcon sx={{ color: 'primary.main', fontSize: 32, mr: 1.5 }} />
            <Typography variant="h5" component="div" sx={{ fontWeight: 800, letterSpacing: '-0.5px' }}>
              Fin<span style={{ color: '#00E676' }}>Stress</span> AI
            </Typography>
          </Box>
          <Box>
            <Button component={Link} to="/" sx={navItemStyle('/')}>
              Accueil
            </Button>
            <Button component={Link} to="/predict" sx={navItemStyle('/predict')}>
              Prédiction
            </Button>
          </Box>
        </Toolbar>
      </Container>
    </AppBar>
  );
}