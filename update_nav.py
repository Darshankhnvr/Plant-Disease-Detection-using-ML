import os
import re

templates_dir = r"c:\Users\admin\Phase Project\PDD\Plant-Disease-Detection-using-ML\templates"

nav_links_content = """        <div class="nav-links">
          <a href="/" class="nav-link">Home</a>
          <a href="/disease-tracker" class="nav-link">Disease Tracker</a>
          <a href="/analytics-dashboard" class="nav-link">Analytics</a>
          <a href="/weather-dashboard" class="nav-link">Weather</a>
          <a href="/#about" class="nav-link">About</a>
          <a href="/#features" class="nav-link">Features</a>
          <a href="/#supported-diseases" class="nav-link">Diseases</a>
        </div>"""

pattern = re.compile(r'<div class="nav-links">.*?</div>', re.DOTALL)

for filename in os.listdir(templates_dir):
    if filename.endswith(".html"):
        filepath = os.path.join(templates_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_content = pattern.sub(nav_links_content, content)
        
        # If it's home.html, we also want to add the Weather Dashboard to quick actions
        if filename == "home.html":
            # Quick actions are in <div class="actions-grid">
            # We will insert the weather dashboard right after the analytics card.
            weather_action_card = """
                <a href="/weather-dashboard" class="action-card">
                  <div class="action-icon">
                    <i class="fas fa-cloud-sun"></i>
                  </div>
                  <div class="action-content">
                    <h3>Weather Forecast</h3>
                    <p>Plantation conditions & risk</p>
                  </div>
                </a>
              </div>"""
            # Replace the end of actions-grid
            content_with_weather = new_content.replace('</div>\n            </div>\n          </div>', weather_action_card + '\n            </div>\n          </div>', 1)
            # Actually, let's just do a regex replace to insert before the closing div of actions-grid
            actions_grid_pattern = re.compile(r'(<div class="actions-grid">.*?)(              </div>\s+</div>\s+</div>)', re.DOTALL)
            
            def insert_weather(match):
                if '/weather-dashboard' not in match.group(1):
                    return match.group(1) + """
                <a href="/weather-dashboard" class="action-card">
                  <div class="action-icon">
                    <i class="fas fa-cloud-sun"></i>
                  </div>
                  <div class="action-content">
                    <h3>Weather Forecast</h3>
                    <p>Plantation conditions & risk</p>
                  </div>
                </a>\n""" + match.group(2)
                return match.group(0)
            
            new_content = actions_grid_pattern.sub(insert_weather, new_content)

        if new_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Updated {filename}")
