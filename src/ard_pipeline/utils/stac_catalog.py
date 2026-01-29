import logging
import planetary_computer
import pystac_client
from pystac_client.stac_api_io import StacApiIO
from urllib3 import Retry
import time

# TODO: Make this changeable...
def http_to_s3_url(http_url):
    """Convert a USGS HTTP URL to an S3 URL"""
    s3_url = http_url.replace(
        "https://landsatlook.usgs.gov/data", "s3://usgs-landsat"
    ).rstrip(":1")
    return s3_url


def connect_to_catalog(catalog_endpoint="landsatlook"):
    # Open a client 
    retry = Retry(
        total=10,
        backoff_factor=1,
        status_forcelist=[404, 502, 503, 504],
        allowed_methods=None, # {*} for CORS
    )
        
    stac_api_io = StacApiIO(max_retries=retry)

    logging.info("Initialize the STAC client")
    if catalog_endpoint=='planetary_computer':
        # Planetary Computer
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
            stac_io=stac_api_io
        )
    elif catalog_endpoint=='earth_search':
        # AWS
        catalog = pystac_client.Client.open(
            "https://earth-search.aws.element84.com/v1",
            stac_io=stac_api_io
            )
    elif catalog_endpoint=='landsatlook':
        catalog = pystac_client. Client.open(
            "https://landsatlook.usgs.gov/stac-server/",
            stac_io=stac_api_io
        )
    else:
        logging.error("You must provide a catalog endpoint alias, available [planetary_computer, earth_search, landsatlook]")

    return catalog

def open_and_search_landsat_in_stac_catalog_single_date(roi_bbox, date, 
                                                        catalog_endpoint="landsatlook",
                                                        collection="landsat-c2l2-sr",
                                                        known_id=None,
                                                        max_retries=2,
                                                        retry_delay=1):
    attempt = 0
    items = None

    while attempt < max_retries:
        try:
            catalog = connect_to_catalog(catalog_endpoint=catalog_endpoint)

            # Access the collection and perform a search
            search = catalog.search(
                collections=collection,
                bbox=roi_bbox,
                datetime="/".join((str(date), str(date))),
                query={
                    "eo:cloud_cover": {"lt": 90},
                    "landsat:cloud_cover_land": {"lt": 90},
                },
            )
            items = search.item_collection()

            if items and known_id:
                wanted_item = [w for w in items if w.properties['landsat:scene_id']==known_id][0]
                if wanted_item:
                    logging.info(f"Found angles for {known_id} on attempt {attempt + 1}.")
                    return wanted_item
                else:
                    items = None  
            elif items and len(items) > 0:
                try:
                    logging.info(f"Found items on attempt {attempt + 1}.")
                    return items  # Return items if valid results are found
                except Exception as e:
                    items = None
            else:
                logging.warning(f"No items found on attempt {attempt + 1}. Retrying...")

        except Exception as e:
            logging.warning(f"Error during STAC search: {e}. Attempt {attempt + 1} of {max_retries}. Retrying...")
        
        attempt += 1
        time.sleep(retry_delay)  # Wait before retrying

    logging.warning("Maximum retries reached. No items found.")
    return None


def open_and_search_landsat_in_stac_catalog(roi_bbox, start_year, end_year):
    # 1.4) Open a client pointing to the AWS or Microsoft Planetary Computer data catalog
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[404, 502, 503, 504],
        allowed_methods=None, # {*} for CORS
    )
        
    stac_api_io = StacApiIO(max_retries=retry)

    # Planetary Computer
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
        stac_io=stac_api_io
    )

    # # AWS
    # catalog = pystac_client.Client.open("https://earth-search.aws.element84.com/v1",
    #                  stac_io=stac_api_io)

    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=roi_bbox,
        # intersects=roi.geometry, #bbox or intersects - not both
        datetime="/".join((str(start_year), str(end_year))),  # '2016/2020'
        query={
            "eo:cloud_cover": {"lt": 90},
            "landsat:cloud_cover_land": {"lt": 90},
            # "platform": {"in": ["landsat-8", "landsat-9"]}
            },
    )

    items = search.item_collection()

    return items


def get_year_group(year):
    if 2000 <= year <= 2024:
        return (year // 5) * 5
    else:
        raise ValueError("Year out of range (2000-2024)")