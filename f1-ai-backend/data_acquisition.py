"""
F1 Data Acquisition Module
Fetches historical race data from Ergast Motor Racing API
"""

import pandas as pd
import requests
import os
import time
from datetime import datetime
from typing import List, Dict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class F1DataAcquisition:
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        self.base_url = "http://ergast.com/api/f1"
        self.consolidated_file = "data/all_f1_results.csv"
        
        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs("data", exist_ok=True)
        
    def fetch_year_results(self, year: int) -> pd.DataFrame:
        """
        Fetch all race results for a specific year from Ergast API
        """
        url = f"{self.base_url}/{year}/results.csv?limit=1000"
        csv_file = f"{self.data_dir}/f1_results_{year}.csv"
        
        try:
            logger.info(f"Fetching data for year {year}...")
            
            # Download CSV
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save raw CSV
            with open(csv_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            # Read into DataFrame
            df = pd.read_csv(csv_file)
            
            # Add season column
            df['season'] = year
            
            logger.info(f"Successfully fetched {len(df)} records for {year}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for year {year}: {e}")
            return pd.DataFrame()
    
    def fetch_all_years(self, start_year: int = 2011, end_year: int = None) -> pd.DataFrame:
        """
        Fetch race results for all years from start_year to current year
        """
        if end_year is None:
            end_year = datetime.now().year
        
        all_data = []
        
        for year in range(start_year, end_year + 1):
            df = self.fetch_year_results(year)
            if not df.empty:
                all_data.append(df)
            
            # Be respectful to API - add delay
            time.sleep(1)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Combined {len(combined_df)} total records from {start_year}-{end_year}")
            return combined_df
        else:
            return pd.DataFrame()
    
    def standardize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize the DataFrame schema to consistent column names
        """
        # Map Ergast columns to our standard schema
        column_mapping = {
            'raceId': 'race_id',
            'year': 'season',
            'round': 'round',
            'circuitId': 'circuit_id',
            'name': 'grand_prix',
            'date': 'date',
            'time': 'race_time_start',
            'driverId': 'driver_id',
            'code': 'driver_code',
            'forename': 'driver_forename',
            'surname': 'driver_surname',
            'constructorId': 'constructor_id',
            'constructor': 'constructor',
            'grid': 'grid_position',
            'position': 'finishing_position',
            'positionText': 'position_text',
            'points': 'points_earned',
            'laps': 'laps_completed',
            'time_y': 'race_time',
            'fastestLap': 'fastest_lap',
            'rank': 'fastest_lap_rank',
            'fastestLapTime': 'fastest_lap_time',
            'fastestLapSpeed': 'fastest_lap_speed',
            'statusId': 'status_id',
            'status': 'status'
        }
        
        # Rename columns that exist
        df_renamed = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Create driver full name
        if 'driver_forename' in df_renamed.columns and 'driver_surname' in df_renamed.columns:
            df_renamed['driver'] = df_renamed['driver_forename'] + ' ' + df_renamed['driver_surname']
        
        # Convert data types
        df_renamed = self._convert_data_types(df_renamed)
        
        return df_renamed
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert columns to appropriate data types
        """
        # Convert numeric columns
        numeric_columns = [
            'season', 'round', 'grid_position', 'finishing_position', 
            'points_earned', 'laps_completed', 'fastest_lap_rank'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Handle position text (convert '\\N' to None for DNF)
        if 'finishing_position' in df.columns:
            df['finishing_position'] = df['finishing_position'].replace('\\N', None)
            df['finishing_position'] = pd.to_numeric(df['finishing_position'], errors='coerce')
        
        return df
    
    def consolidate_data(self, start_year: int = 2011, end_year: int = None) -> pd.DataFrame:
        """
        Main method to fetch, standardize, and consolidate all race data
        """
        logger.info("Starting F1 data consolidation...")
        
        # Fetch all years
        raw_data = self.fetch_all_years(start_year, end_year)
        
        if raw_data.empty:
            logger.error("No data fetched!")
            return pd.DataFrame()
        
        # Standardize schema
        standardized_data = self.standardize_schema(raw_data)
        
        # Save consolidated data
        standardized_data.to_csv(self.consolidated_file, index=False)
        logger.info(f"Consolidated data saved to {self.consolidated_file}")
        
        # Print summary
        self._print_data_summary(standardized_data)
        
        return standardized_data
    
    def _print_data_summary(self, df: pd.DataFrame):
        """
        Print summary statistics of the consolidated data
        """
        logger.info("=== DATA SUMMARY ===")
        logger.info(f"Total records: {len(df):,}")
        
        if len(df) == 0:
            logger.info("No data available for summary")
            return
            
        logger.info(f"Years covered: {df['season'].min()} - {df['season'].max()}")
        logger.info(f"Unique drivers: {df['driver'].nunique()}")
        logger.info(f"Unique constructors: {df['constructor'].nunique() if 'constructor' in df.columns else 'N/A'}")
        logger.info(f"Total races: {df.groupby(['season', 'round']).ngroups}")
        
        # Show sample data
        logger.info("\n=== SAMPLE DATA ===")
        print(df[['season', 'round', 'grand_prix', 'driver', 'constructor', 
                 'grid_position', 'finishing_position', 'status']].head(10))
                 
    def acquire_multi_year_data(self, years: List[int], force_refresh: bool = False) -> pd.DataFrame:
        """
        Acquire F1 data for multiple years using the professional pipeline.
        This method is used by the professional API endpoints.
        
        Args:
            years: List of years to acquire data for
            force_refresh: Whether to force refresh existing data
            
        Returns:
            Consolidated DataFrame with all requested years
        """
        logger.info(f"Starting multi-year data acquisition for years: {years}")
        
        if not years:
            logger.warning("No years specified for data acquisition")
            return pd.DataFrame()
        
        # Check if consolidated data already exists and is not empty
        if os.path.exists(self.consolidated_file) and not force_refresh:
            try:
                existing_data = pd.read_csv(self.consolidated_file)
                if len(existing_data) > 0:
                    logger.info(f"Using existing consolidated data with {len(existing_data)} records")
                    return existing_data
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                logger.info("Existing data file is empty or corrupt, will create new data")
                pass
        
        all_data = []
        successful_years = []
        
        for year in years:
            try:
                logger.info(f"Acquiring data for year {year}...")
                year_data = self.fetch_year_results(year)
                year_data = self.standardize_schema(year_data)
                all_data.append(year_data)
                successful_years.append(year)
                
                # Small delay between API calls to be respectful
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Failed to acquire data for year {year}: {e}")
                continue
        
        if not all_data:
            logger.warning("No data acquired from external API, creating sample data for demonstration")
            # Create sample data for demonstration when external API is not available
            return self._create_sample_data(years)
        
        # Combine all years
        consolidated_data = pd.concat(all_data, ignore_index=True)
        consolidated_data = self._convert_data_types(consolidated_data)
        
        # Save consolidated data
        consolidated_data.to_csv(self.consolidated_file, index=False)
        logger.info(f"Data acquisition complete! {len(consolidated_data)} records saved to {self.consolidated_file}")
        
        self._print_data_summary(consolidated_data)
        
        return consolidated_data
    
    def _create_sample_data(self, years: List[int]) -> pd.DataFrame:
        """
        Create sample F1 data for demonstration when external API is not available
        """
        logger.info("Creating sample F1 data for demonstration purposes")
        
        drivers = ["Max Verstappen", "Charles Leclerc", "Lando Norris", "George Russell", 
                  "Carlos Sainz", "Lewis Hamilton", "Sergio Perez", "Fernando Alonso"]
        constructors = ["Red Bull", "Ferrari", "McLaren", "Mercedes", "Aston Martin", "Alpine"]
        circuits = ["Monaco", "Silverstone", "Monza", "Spa-Francorchamps", "Suzuka"]
        
        sample_data = []
        
        for year in years:
            for round_num in range(1, 6):  # 5 races per year for demo
                for pos, driver in enumerate(drivers, 1):
                    sample_data.append({
                        'season': year,
                        'round': round_num,
                        'grand_prix': circuits[(round_num - 1) % len(circuits)],
                        'driver': driver,
                        'constructor': constructors[pos % len(constructors)],
                        'grid_position': pos,
                        'finishing_position': pos + (pos % 3),  # Some variation
                        'status': 'Finished' if pos <= 6 else 'DNF',
                        'points': max(0, 25 - (pos - 1) * 3) if pos <= 10 else 0,
                        'fastest_lap': pos == 1,
                        'time': f"1:{55 + pos}:{20 + pos}.{100 + pos}",
                        'laps': 70 - (pos % 5)
                    })
        
        df = pd.DataFrame(sample_data)
        
        # Save sample data
        df.to_csv(self.consolidated_file, index=False)
        logger.info(f"Sample data created with {len(df)} records and saved to {self.consolidated_file}")
        
        return df

# Example usage and testing
if __name__ == "__main__":
    # Initialize data acquisition
    data_fetcher = F1DataAcquisition()
    
    # Fetch and consolidate data (this will take several minutes)
    consolidated_data = data_fetcher.consolidate_data(start_year=2020, end_year=2024)  # Start with recent years for testing
    
    print(f"Data consolidation complete! {len(consolidated_data)} records saved.")
