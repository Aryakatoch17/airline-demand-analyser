import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import random
import google.generativeai as genai

# Handle secrets for Streamlit Cloud deployment
try:
    # Try to load from Streamlit secrets first (for cloud deployment)
    AVIATIONSTACK_API_KEY = st.secrets["AVIATIONSTACK_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    # Fallback to environment variables (for local development)
    AVIATIONSTACK_API_KEY = os.getenv('AVIATIONSTACK_API_KEY')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Australian airports for filtering (expanded list)
AUSTRALIAN_AIRPORTS = {
    'SYD': 'Sydney', 'MEL': 'Melbourne', 'BNE': 'Brisbane', 'PER': 'Perth',
    'ADL': 'Adelaide', 'DRW': 'Darwin', 'CNS': 'Cairns', 'GLD': 'Gold Coast',
    'HBA': 'Hobart', 'CBR': 'Canberra', 'TSV': 'Townsville', 'ROK': 'Rockhampton',
    'MKY': 'Mackay', 'BDB': 'Bundaberg', 'HTI': 'Hamilton Island', 'PPP': 'Proserpine',
    'LST': 'Launceston', 'DPO': 'Devonport', 'ARM': 'Armidale', 'TMW': 'Tamworth'
}

class AirlineDataFetcher:
    def __init__(self):
        self.base_url = "http://api.aviationstack.com/v1"
        self.api_key = AVIATIONSTACK_API_KEY

    def get_flight_data(self, limit=100, dep_iata=None, arr_iata=None):
        """Fetch real-time flight data from AviationStack API"""
        url = f"{self.base_url}/flights"
        params = {
            'access_key': self.api_key,
            'limit': min(limit, 100)
        }
        
        # Add filtering parameters if provided
        if dep_iata:
            params['dep_iata'] = dep_iata
        if arr_iata:
            params['arr_iata'] = arr_iata
            
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching flight data: {e}")
            return None

    def get_australian_flights(self, limit=100):
        """Fetch flights specifically for Australian routes"""
        all_flights = []
        australian_codes = list(AUSTRALIAN_AIRPORTS.keys())
        
        # Try to get flights from major Australian airports
        major_airports = ['SYD', 'MEL', 'BNE', 'PER']
        
        for airport in major_airports:
            if len(all_flights) >= limit:
                break
                
            # Get departures from this airport
            flights_data = self.get_flight_data(limit=25, dep_iata=airport)
            if flights_data and 'data' in flights_data:
                all_flights.extend(flights_data['data'])
            
            # Small delay to avoid rate limiting
            import time
            time.sleep(0.5)
        
        # If we still don't have enough data, try arrivals
        if len(all_flights) < 20:
            for airport in major_airports[:2]:  # Just try SYD and MEL for arrivals
                flights_data = self.get_flight_data(limit=25, arr_iata=airport)
                if flights_data and 'data' in flights_data:
                    all_flights.extend(flights_data['data'])
                import time
                time.sleep(0.5)
        
        return {'data': all_flights[:limit]}

    def simulate_price_data(self):
        """Simulate price data for demonstration purposes"""
        routes = ['SYD-MEL', 'SYD-BNE', 'MEL-PER', 'SYD-ADL', 'BNE-PER', 'SYD-PER', 'MEL-BNE']
        price_data = []
        for route in routes:
            for day_offset in range(1, 31):
                date = datetime.now() + timedelta(days=day_offset)
                base_price = random.randint(150, 700)
                if date.weekday() >= 5:
                    base_price *= 1.15
                if date.month in [12, 1, 7]:
                    base_price *= 1.25
                price_data.append({
                    'route': route, 'date': date.strftime('%Y-%m-%d'),
                    'price': round(base_price), 'day_of_week': date.strftime('%A'),
                    'month': date.strftime('%B')
                })
        return pd.DataFrame(price_data)

class DataProcessor:
    @staticmethod
    def process_flight_data(raw_data):
        if not raw_data or 'data' not in raw_data:
            return pd.DataFrame()
        
        flights = []
        for flight in raw_data['data']:
            if flight and 'departure' in flight and 'arrival' in flight:
                dep_iata = flight.get('departure', {}).get('iata', '')
                arr_iata = flight.get('arrival', {}).get('iata', '')
                
                # More robust data extraction
                flight_info = {
                    'flight_number': flight.get('flight', {}).get('number', 'N/A'),
                    'airline': flight.get('airline', {}).get('name', 'Unknown'),
                    'departure_airport': flight.get('departure', {}).get('airport', 'Unknown'),
                    'arrival_airport': flight.get('arrival', {}).get('airport', 'Unknown'),
                    'departure_iata': dep_iata,
                    'arrival_iata': arr_iata,
                    'status': flight.get('flight_status', 'Unknown'),
                    'departure_time': flight.get('departure', {}).get('scheduled', 'Unknown')
                }
                
                # Check if it's an Australian route (more comprehensive check)
                is_aus_dep = dep_iata in AUSTRALIAN_AIRPORTS
                is_aus_arr = arr_iata in AUSTRALIAN_AIRPORTS
                is_aus_dep_name = any(aus_city.lower() in flight_info['departure_airport'].lower() 
                                    for aus_city in AUSTRALIAN_AIRPORTS.values())
                is_aus_arr_name = any(aus_city.lower() in flight_info['arrival_airport'].lower() 
                                    for aus_city in AUSTRALIAN_AIRPORTS.values())
                
                flight_info['is_australian_route'] = (is_aus_dep or is_aus_arr or 
                                                    is_aus_dep_name or is_aus_arr_name)
                
                flights.append(flight_info)
        
        df = pd.DataFrame(flights)
        
        if not df.empty:
            # Add time-based columns
            df['departure_datetime'] = pd.to_datetime(df['departure_time'], errors='coerce')
            df['departure_hour'] = df['departure_datetime'].dt.hour
            df['departure_day'] = df['departure_datetime'].dt.day_name()
            df['departure_month'] = df['departure_datetime'].dt.month_name()
        
        return df

    @staticmethod
    def analyze_popular_routes(df, australia_only=False):
        if df.empty:
            return pd.DataFrame()
        
        # Filter for Australian routes if requested
        if australia_only:
            df_filtered = df[df['is_australian_route'] == True]
            if df_filtered.empty:
                # If no Australian routes found, return empty with message
                return pd.DataFrame()
        else:
            df_filtered = df
        
        if df_filtered.empty:
            return pd.DataFrame()
        
        route_counts = df_filtered.groupby(['departure_airport', 'arrival_airport']).size().reset_index(name='frequency')
        route_counts['route'] = route_counts['departure_airport'] + ' → ' + route_counts['arrival_airport']
        return route_counts.sort_values('frequency', ascending=False).head(10)

    @staticmethod
    def analyze_airline_demand(df):
        if df.empty:
            return pd.DataFrame()
        airline_counts = df['airline'].value_counts().reset_index()
        airline_counts.columns = ['airline', 'flight_count']
        return airline_counts.head(10)

    @staticmethod
    def analyze_price_trends(price_df):
        if price_df.empty:
            return pd.DataFrame(), pd.DataFrame()
        avg_price_by_route = price_df.groupby('route')['price'].mean().reset_index()
        avg_price_by_day = price_df.groupby('day_of_week')['price'].mean().reset_index()
        return avg_price_by_route, avg_price_by_day

class InsightGenerator:
    def __init__(self):
        """Initialize Gemini AI"""
        self.model = None
        if GEMINI_API_KEY:
            try:
                self.model = genai.GenerativeModel('gemini-1.5-flash')
            except Exception as e:
                st.warning(f"Failed to initialize Gemini: {e}")
    
    def generate_insights(self, data_summary):
        """Generate insights using Google Gemini"""
        if not self.model:
            return "Gemini API key not configured. Please add GEMINI_API_KEY to your .env file."
        
        try:
            prompt = f"""
            As a business analyst specializing in the Australian airline and hostel industry, 
            analyze this flight market data and provide strategic insights:

            FLIGHT DATA ANALYSIS:
            {data_summary}

            Please provide 4-5 key insights focusing on:
            1. **Route Popularity**: Which routes show highest demand for hostel planning
            2. **Market Opportunities**: Airlines and destinations with growth potential
            3. **Seasonal Patterns**: Travel trends affecting hostel occupancy
            4. **Business Recommendations**: Actionable advice for hostel operators
            5. **Partnership Opportunities**: Airlines or routes for potential collaborations

            Format your response with clear headers and bullet points.
            Keep insights practical and specific to the Australian travel market.
            """
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"Unable to generate AI insights: {e}"

def main():
    st.set_page_config(page_title="Australian Airline Market Intelligence", page_icon="✈️", layout="wide")
    
    st.title("✈️ Australian Airline Market Intelligence Dashboard")
    st.markdown("*Real-time flight analysis for hostel business insights - Powered by Google Gemini AI*")
    
    # Information box
    st.info("""
    📊 **How to use this app:**
    1. Adjust analysis settings in the sidebar
    2. View real-time Australian flight trends and market insights
    3. Generate AI-powered business recommendations
    4. Export your analysis for further use
    """)

    # Initialize classes
    fetcher = AirlineDataFetcher()
    processor = DataProcessor()
    insight_gen = InsightGenerator()

    # Sidebar controls
    st.sidebar.header("🔧 Analysis Settings")
    
    # Geographic focus
    st.sidebar.subheader("Geographic Focus")
    australia_focus = st.sidebar.checkbox(
        "Filter for Australian routes only", 
        value=True,
        help="Show only flights involving Australian airports"
    )
    
    # Data controls
    st.sidebar.subheader("Data Settings")
    data_limit = st.sidebar.slider("Number of flights to analyze", 50, 100, 100,
                                  help="Maximum 100 flights due to API limitations")
    
    # Analysis options
    st.sidebar.subheader("Analysis Options")
    show_price = st.sidebar.checkbox("Show Price Trend Analysis", value=True)
    show_insights = st.sidebar.checkbox("Enable AI Insights (Gemini)", value=True)
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.success("Data cache cleared!")

    # API key checks
    if not AVIATIONSTACK_API_KEY:
        st.error("⚠️ AviationStack API key not found. Please add it to your .env file.")
        st.code("AVIATIONSTACK_API_KEY=your_api_key_here")
        return
    
    if show_insights and not GEMINI_API_KEY:
        st.warning("⚠️ Gemini API key not found. AI insights will be disabled.")
        st.code("GEMINI_API_KEY=your_gemini_api_key_here")

    # Main data fetching and analysis - Always use Australia-focused data
    with st.spinner("Fetching Australian flight data..."):
        raw_data = fetcher.get_australian_flights(limit=data_limit)

    if raw_data:
        df = processor.process_flight_data(raw_data)

        if not df.empty:
            # Key Metrics
            st.header("📊 Flight Data Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Flights", len(df))
            
            with col2:
                st.metric("Unique Airlines", df['airline'].nunique())
            
            with col3:
                australian_routes = df['is_australian_route'].sum()
                st.metric("Australian Routes", australian_routes)
            
            with col4:
                active_flights = len(df[df['status'] == 'active'])
                st.metric("Active Flights", active_flights)

            # Show data source info
            st.info(f"📡 **Data Source**: Australia-Focused Data | **Australian Routes Found**: {australian_routes} out of {len(df)} total flights")

            # Popular Routes Analysis
            st.header("🗺️ Popular Routes Analysis")
            popular_routes = processor.analyze_popular_routes(df, australia_only=australia_focus)
            
            if not popular_routes.empty:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    title = "Top Routes in Australia" if australia_focus else "Top Routes"
                    fig_routes = px.bar(
                        popular_routes, 
                        x='frequency', 
                        y='route', 
                        orientation='h',
                        title=title,
                        labels={'frequency': 'Number of Flights', 'route': 'Route'},
                        color='frequency',
                        color_continuous_scale='Blues'
                    )
                    fig_routes.update_layout(height=400)
                    st.plotly_chart(fig_routes, use_container_width=True)
                
                with col2:
                    st.subheader("Route Details")
                    st.dataframe(popular_routes[['route', 'frequency']])
            else:
                if australia_focus:
                    st.warning("🔍 No Australian routes found in current dataset. Try disabling the Australian filter or refreshing the data.")
                    st.info("💡 **Tip**: The API might not have Australian flights in the current sample. Try refreshing the data.")
                else:
                    st.warning("No route data available.")

            # Airline Market Share
            st.header("✈️ Airline Market Analysis")
            airline_demand = processor.analyze_airline_demand(df)
            
            if not airline_demand.empty:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    fig_pie = px.pie(
                        airline_demand.head(8), 
                        values='flight_count', 
                        names='airline',
                        title="Market Share by Airline"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    fig_bar = px.bar(
                        airline_demand, 
                        x='airline', 
                        y='flight_count',
                        title="Flight Count by Airline",
                        color='flight_count',
                        color_continuous_scale='Viridis'
                    )
                    fig_bar.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_bar, use_container_width=True)

            st.header("📈 Flight Status Overview")
            status_counts = df['status'].value_counts()
            
            fig_status = px.bar(
                x=status_counts.index, 
                y=status_counts.values,
                title="Flight Status Distribution",
                labels={'x': 'Status', 'y': 'Number of Flights'},
                color=status_counts.values,
                color_continuous_scale='RdYlBu'
            )
            st.plotly_chart(fig_status, use_container_width=True)

            # Price Trend Analysis
            if show_price:
                st.header("💰 Price Trend Analysis")
                st.info("📝 Note: Price data is simulated for demonstration. In production, this would use real pricing APIs.")
                
                price_df = fetcher.simulate_price_data()
                avg_price_by_route, avg_price_by_day = processor.analyze_price_trends(price_df)

                col1, col2 = st.columns([1, 1])
                
                with col1:
                    fig_price_route = px.bar(
                        avg_price_by_route, 
                        x='route', 
                        y='price',
                        title="Average Flight Price by Route (AUD)",
                        color='price',
                        color_continuous_scale='Reds'
                    )
                    st.plotly_chart(fig_price_route, use_container_width=True)
                
                with col2:
                    fig_price_day = px.bar(
                        avg_price_by_day, 
                        x='day_of_week', 
                        y='price',
                        title="Average Flight Price by Day of Week (AUD)",
                        color='price',
                        color_continuous_scale='Greens'
                    )
                    st.plotly_chart(fig_price_day, use_container_width=True)

            # AI-Generated Insights
            if show_insights:
                st.header("🤖 AI-Generated Business Insights")
                
                if st.button("🔮 Generate Strategic Insights", type="primary"):
                    if not GEMINI_API_KEY:
                        st.warning("Gemini API key not configured. Please add GEMINI_API_KEY to your .env file.")
                    else:
                        with st.spinner("Analyzing data with Google Gemini AI..."):
                            data_summary = f"""
                            FLIGHT ANALYSIS SUMMARY:
                            - Total flights analyzed: {len(df)}
                            - Australian routes: {australian_routes}
                            - Data source: Australia-Focused Data
                            - Analysis focus: {'Australia-specific' if australia_focus else 'All available data'}
                            
                            TOP AIRLINES:
                            {airline_demand.head(5).to_dict('records') if not airline_demand.empty else 'No airline data'}
                            
                            TOP ROUTES:
                            {popular_routes.head(5).to_dict('records') if not popular_routes.empty else 'No route data'}
                            
                            FLIGHT STATUS DISTRIBUTION:
                            {status_counts.to_dict()}
                            
                            MARKET CONTEXT:
                            - Focus on Australian hostel business opportunities
                            - Backpacker and budget travel market
                            - Seasonal tourism patterns
                            """
                            
                            insights = insight_gen.generate_insights(data_summary)
                            st.markdown(insights)

            # Data Export
            st.header("💾 Export Your Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            with col1:
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📄 Download Flight Data",
                    data=csv_data,
                    file_name=f"flight_data_{timestamp}.csv",
                    mime="text/csv"
                )
            
            with col2:
                if not popular_routes.empty:
                    routes_csv = popular_routes.to_csv(index=False)
                    st.download_button(
                        label="🗺️ Download Route Analysis",
                        data=routes_csv,
                        file_name=f"route_analysis_{timestamp}.csv",
                        mime="text/csv"
                    )
            
            with col3:
                if not airline_demand.empty:
                    airline_csv = airline_demand.to_csv(index=False)
                    st.download_button(
                        label="✈️ Download Airline Analysis",
                        data=airline_csv,
                        file_name=f"airline_analysis_{timestamp}.csv",
                        mime="text/csv"
                    )

            # Raw Data Explorer
            with st.expander("📋 Explore Raw Flight Data"):
                st.subheader("Flight Data Table")
                search_term = st.text_input("🔍 Search flights (airline, airport, status):")
                
                if search_term:
                    mask = df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
                    filtered_df = df[mask]
                    st.write(f"Found {len(filtered_df)} flights matching '{search_term}'")
                    st.dataframe(filtered_df, use_container_width=True)
                else:
                    st.dataframe(df, use_container_width=True)
                
                # Data summary
                st.subheader("Data Summary")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Dataset Information:**")
                    st.write(f"- Total flights: {len(df)}")
                    st.write(f"- Australian routes: {australian_routes}")
                    st.write(f"- Unique airlines: {df['airline'].nunique()}")
                
                with col2:
                    st.write("**Status Distribution:**")
                    for status, count in status_counts.items():
                        st.write(f"- {status}: {count}")

        else:
            st.warning("No flight data available to analyze.")
    
    else:
        st.error("❌ Failed to fetch flight data.")

    # Footer
    st.markdown("---")
    st.markdown("""
    **About this app:**
    - Built with Streamlit and Python
    - Uses AviationStack API for real-time flight data
    - Powered by Google Gemini AI for insights
    - Designed for Australian hostel and travel industry analysis
    """)

if __name__ == '__main__':
    main()
