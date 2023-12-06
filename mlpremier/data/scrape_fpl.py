"""
Scrape raw data from vaastv's FPL repo. 
"""

import pandas as pd
import numpy as np
from fnmatch import fnmatch
import requests
import os
from typing import List
import concurrent.futures
import argparse
from dotenv import load_dotenv
from tqdm import tqdm

def get_csv_recursive(github_token:str, repo:str, path:str)->List[str]:
        """ 
        Recursively retrieve .csv download paths from the specified github repo 
        and data path.

        :param str github_token: Github API token with public_repo download access.
        :param str repo: GH repo in the form gituser/repo
        :param str path: path within repo to desired data folder /path/to/data

        :returns: A list of recursively retrieved download urls for the csv 
            files in the provided repo and folder path
        :rtype: List[str]
        """
        headers = {
            "Authorization": f"token {github_token}"
        }
        url = f"https://api.github.com/repos/{repo}/contents/{path}"
        response = requests.get(url, headers=headers)
        response_data = response.json()

        csv_files = []
        for item in response_data:
            if item["type"] == "file" and item["name"].endswith(".csv"):
                csv_files.append(item["download_url"])
            elif item["type"] == "dir":
                # Recursively get CSV files in subdirectories.
                subfolder_path = os.path.join(path, item["name"])
                csv_files.extend(get_csv_recursive(github_token, repo, subfolder_path))
        return csv_files

def identify_fpl_data(github_token:str,
                      fpl_repo:str, 
                      season:str = "2021-22",
                      retrieve:str = 'players',
                      verbose:bool = False) -> List[str]:
    """ 
    Identify FPL data by player for the provided season.

    :param str github_token: Github API token with public_repo download access.
    :param str fpl_repo: FPL data repo path in form gituser/repo
    :param str season: Which season of EPL data to download. Should 
        follow the format 20XX-X(X+1), with the earliest data available being
        2016-17.
    :param str retrieve: Retrieval type ('players' or 'meta'). Players retrieves 
        csv files for each player across gameweeks. Meta retreives the full 
        gameweek data for each gameweek (from which metadata can be extracted 
        about player positions and teams).

    :return: List of CSV files to download from desired repo section.
    :rtype: List[str]
    """
   
    # Initialize headers with the personal access token.
    headers = {
        "Authorization": f"token {github_token}"
    }

    if fpl_repo is None:
        fpl_repo = os.path.join("vaastav", "Fantasy-Premier-League")

    if retrieve == 'players':
        data_folder_path = os.path.join("data")
        season_path = os.path.join(f"{season}", "players") 
        path = os.path.join(data_folder_path, season_path)
        
        if verbose:
            print("Retrieving list of CSV download URLs for requested repo folder.")
        csv_urls = get_csv_recursive(github_token, fpl_repo, path)
        
        print("CSVS Identified.")
        
    elif retrieve == 'meta':
         #get full gameweek CSVs for player metadata
        data_folder_path = os.path.join("data")
        metadata_path = os.path.join(f"{season}", "gws") 
        path = os.path.join(data_folder_path, metadata_path)

        if verbose:
                print("Retrieving metadata (full GW CSVs) URLS.")
        csv_urls = get_csv_recursive(github_token, fpl_repo, path)
        csv_urls = [url for url in csv_urls if fnmatch(url, '*gw*.csv')]

    else:
         raise Exception(("Invalid 'retrieve' retrieval type specified." 
                          "Choose one of 'players' or 'meta'"))

    return csv_urls

def download_fpl_data(github_token:str,
                      csv_urls:List[str],
                      save_dir:str = "raw_data",
                      retrieve:str = 'players',
                      verbose:bool = False) -> None:
    """ 
    Download FPL data by player for the provided season.

    :param str github_token: Github API token with public_repo download access.
    :param List[str] csv_urls: csv urls to download
    :param str save_dir: Directory to save data in.
    :param str retrieve: Retrieval type ('players' or 'meta'). Players retrieves 
        csv files for each player across gameweeks. Meta retreives the full 
        gameweek data for each gameweek (from which metadata can be extracted 
        about player positions and teams). In this function, this only changes
        the save directory behavior since expected data changes dir structure.

    :return: None
    :rtype: None
    """
    # Initialize headers with the personal access token.
    headers = {
        "Authorization": f"token {github_token}"
    }

    # Download and save CSV files locally.
    for csv_url in tqdm(csv_urls):
        response = requests.get(csv_url, headers=headers)
        
        if response.status_code == 200:

            write_dirs = None
            filename = None
            if retrieve == 'players':
                #construct filename to write to
                dirs = os.path.dirname(csv_url)
                split_dirs = dirs.split(os.path.sep)
                year_dir = split_dirs[-3]
                players_dir = split_dirs[-2] 
                name_dir = split_dirs[-1]
                write_dirs = os.path.join(os.path.abspath(os.path.curdir), 
                                        save_dir, 
                                        year_dir,
                                        players_dir,
                                        name_dir)

                player_filename = os.path.basename(csv_url)

                filename = os.path.join(write_dirs,
                                        player_filename)
            elif retrieve == 'meta':
                 #construct filename to write to
                dirs = os.path.dirname(csv_url)
                split_dirs = dirs.split(os.path.sep)
                year_dir = split_dirs[-2]
                gws_dir = split_dirs[-1] 
                write_dirs = os.path.join(os.path.abspath(os.path.curdir), 
                                        save_dir, 
                                        year_dir,
                                        gws_dir)

                gw_filename = os.path.basename(csv_url)

                filename = os.path.join(write_dirs,
                                        gw_filename)
                
            else:
                raise Exception(("Invalid 'retrieve' retrieval type specified." 
                          "Choose one of 'players' or 'meta'"))
            
            #add directories if needed
            if not os.path.exists(write_dirs):
                os.makedirs(write_dirs)

            #write the data
            if verbose:
                print(f"Downloading CSV File: {filename}...")
            with open(filename, "wb") as f:
                f.write(response.content)

    return

def get_master_team_list(github_token: str,
                         save_dir: str = "clean_data",
                         verbose: bool = False) -> None:
    """ 
    Download FPL master team list from vaastv's FPL Repo.

    :param str github_token: Github API token with public_repo download access.
    :param str save_dir: Directory to save data in.
    :param bool verbose: Whether to print verbose output.

    :return: None
    :rtype: None
    """

    # Define the URL for the master_team_list.csv on vaastav's FPL Repo
    github_repo_url = ("https://raw.githubusercontent.com/vaastav/"
                       "Fantasy-Premier-League/master/data/")
    csv_url = f"{github_repo_url}master_team_list.csv"

    headers = {"Authorization": f"token {github_token}"}
    response = requests.get(csv_url, headers=headers)

    if response.status_code == 200:
        os.makedirs(save_dir, exist_ok=True)

        # Save the raw CSV content to a file in the specified directory
        save_path = os.path.join(save_dir, "master_team_list.csv")
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(response.text)

        if verbose:
            print(f"Master team list saved to: {save_path}")
    else:
        print(f"Failed to download master team list. Status Code: {response.status_code}")

    return


def main():
    """
    Scrape, Clean, Store the requested FPL Data.
    """
    parser = argparse.ArgumentParser(description='Scrape FPL data.')
    parser.add_argument('-s', '--season', type=str, help='Comma-separated list of seasons')
    parser.add_argument('-m', '--master', action='store_true', default=False, help='Download the master team list')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Verbose download file progress.')
    args = parser.parse_args()

    # Load .env file from dir where script is called
    dotenv_path = os.path.join(os.getcwd(), '.env')
    dotenv_path = os.path.join(os.getcwd(), '.env')
    load_dotenv(dotenv_path)

    FPL_REPO = os.path.join("vaastav", "Fantasy-Premier-League")
    RETRIEVE = 'players'
    GH_KEY = os.getenv('GITHUB_TOKEN', None)
    assert GH_KEY is not None, "Github Key for Public Repo Access must be stored in .env"

    # Access the value of the 'season' option
    if args.season:
        seasons = args.season.split(',')
        print(f'Seasons to be downloaded: {seasons}')

        for season in seasons:
            print(f"\n======= Downloading data for season: {season} =========")

            print("\n====== Identifying FPL Data to Download =====")
            csv_urls = identify_fpl_data(GH_KEY, FPL_REPO, season, RETRIEVE, verbose=True)

            print("\n====== Downloading FPL Data =====")
            download_fpl_data(GH_KEY, csv_urls, "raw_data", verbose=True)

            meta_urls = identify_fpl_data(GH_KEY, FPL_REPO, 
                                    season=season, 
                                    retrieve='meta', 
                                    verbose=args.verbose)
            download_fpl_data(GH_KEY, meta_urls, "raw_data", retrieve='meta',verbose=True)
    else:    
        print("No seasons requested.")
    
    if args.master:
        get_master_team_list(GH_KEY, verbose=True)

    print("Done. Quitting.")

if __name__ == '__main__':
    main()