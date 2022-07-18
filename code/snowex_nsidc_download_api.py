from snowex_nsidc_search import * 



#Set NSIDC data access base URL
base_url = 'https://n5eil02u.ecs.nsidc.org/egi/request'

#Set the request mode to asynchronous, "no" processing agent (no subsetting or reformatting services available), and optionally removing metadata delivery

param_dict['request_mode'] = 'async'
param_dict['agent'] = 'NO'
param_dict['INCLUDE_META'] ='N' #optional if you do not wish to receive the associated metadata files with each science file. 

param_string = '&'.join("{!s}={!r}".format(k,v) for (k,v) in param_dict.items()) # Convert param_dict to string
param_string = param_string.replace("'","") # Remove quotes

api_list = [f'{base_url}?{param_string}']
api_request = api_list[0]
print(api_request) # Print API base URL + request parameters

# Start authenticated session with Earthdata Login to allow for data downloads:
def setup_earthdata_login_auth(endpoint: str='urs.earthdata.nasa.gov'):
    netrc_name = "_netrc" if system()=="Windows" else ".netrc"
    try:
        username, _, password = netrc.netrc(file=join(expanduser('~'), netrc_name)).authenticators(endpoint)
    except (FileNotFoundError, TypeError):
        print('Please provide your Earthdata Login credentials for access.')
        print('Your info will only be passed to %s and will not be exposed in Jupyter.' % (endpoint))
        username = input('Username: ')
        password = getpass('Password: ')
    manager = request.HTTPPasswordMgrWithDefaultRealm()
    manager.add_password(None, endpoint, username, password)
    auth = request.HTTPBasicAuthHandler(manager)
    jar = CookieJar()
    processor = request.HTTPCookieProcessor(jar)
    opener = request.build_opener(auth, processor)
    request.install_opener(opener)

setup_earthdata_login_auth(endpoint="urs.earthdata.nasa.gov")


def request_nsidc_data(API_request):
    """
    Performs a data customization and access request from NSIDC's API/
    Creates an output folder in the working directory if one does not already exist.
    
    :API_request: NSIDC API endpoint; see https://nsidc.org/support/how/how-do-i-programmatically-request-data-services for more info
    on how to configure the API request.
    
    """

    path = str(f'{datadir}') # Create an output folder if the folder does not already exist.
    if not os.path.exists(path):
        os.mkdir(path)
        
    base_url = 'https://n5eil02u.ecs.nsidc.org/egi/request'

    
    r = request.urlopen(API_request)
    esir_root = ET.fromstring(r.read())
    orderlist = []   # Look up order ID
    for order in esir_root.findall("./order/"):
        orderlist.append(order.text)
    orderID = orderlist[0]
    statusURL = base_url + '/' + orderID # Create status URL
    print('Order status URL: ', statusURL)
    request_response = request.urlopen(statusURL) # Find order status  
    request_root = ET.fromstring(request_response.read())
    statuslist = []
    for status in request_root.findall("./requestStatus/"):
        statuslist.append(status.text)
    status = statuslist[0]
    while status == 'pending' or status == 'processing': #Continue loop while request is still processing
        print('Job status is ', status,'. Trying again.')
        time.sleep(10)
        loop_response = request.urlopen(statusURL)
        loop_root = ET.fromstring(loop_response.read())
        statuslist = [] #find status
        for status in loop_root.findall("./requestStatus/"):
            statuslist.append(status.text)
        status = statuslist[0]
        if status == 'pending' or status == 'processing':
            continue
    if status == 'complete_with_errors' or status == 'failed': # Provide complete_with_errors error message:
        messagelist = []
        for message in loop_root.findall("./processInfo/"):
            messagelist.append(message.text)
        print('Job status is ', status)
        print('error messages:')
        pprint(messagelist)
    if status == 'complete' or status == 'complete_with_errors':# Download zipped order if status is complete or complete_with_errors
        downloadURL = 'https://n5eil02u.ecs.nsidc.org/esir/' + orderID + '.zip'
        print('Job status is ', status)
        print('Zip download URL: ', downloadURL)
        print('Beginning download of zipped output...')
        zip_response = request.urlopen(downloadURL)
        with zipfile.ZipFile(io.BytesIO(zip_response.read())) as z:
            z.extractall(path)
        print('Download is complete.')
    else: print('Request failed.')
    
    # Clean up Outputs folder by removing individual granule folders 
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            try:
                shutil.move(os.path.join(root, file), path)
            except OSError:
                pass
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    return  


# NOTE: downloads ~ 200MB of CSV files
request_nsidc_data(api_request)
