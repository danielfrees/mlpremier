# Data Processing and Storage for FPL Modeling

`scrape_fpl.ipynb` can be used to scrape player-by-player CSV data for a 
specified FPL season from https://github.com/vaastav/Fantasy-Premier-League/.

`preprocesss_fpl.ipynb` takes the data, adds player position, name, team name, 
separates into folders by different positions (GKP, DEF, MID, FWD)


Do note that I do nothing special to deal with player transfers. This could be 
a good future step to improve data quality, though I suspect it will have very minor
effect on the model performance since *most* players will not transfer teams within 
the premier league mid-season.