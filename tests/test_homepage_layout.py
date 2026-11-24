"""Guard the product's simple default home layout without coupling to live Streamlit."""
from pathlib import Path
import unittest


class HomepageLayoutTests(unittest.TestCase):
    def test_simple_home_sections_and_advanced_mode(self):
        app = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
        home = app.split('if page=="Stock Scores":', 1)[1].split('elif page=="Overview":', 1)[0]
        self.assertIn('st.tabs(["Compare stocks","Discover by sector","My watchlist"])', home)
        self.assertIn('with st.expander("Score history and price chart"', home)
        self.assertIn('st.slider("Momentum weight"', home)
        self.assertIn('x.metric("Your custom Hybrid"', home)
        self.assertIn('st.sidebar.toggle("Advanced research"', app)
        self.assertIn('signal_engine.weighted_hybrid(m,r,w/100.0)', home)
        self.assertLess(home.index('st.slider("Momentum weight"'), home.index('st.tabs(['))


if __name__ == "__main__":
    unittest.main()
