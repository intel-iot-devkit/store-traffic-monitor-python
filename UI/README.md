# Store traffic monitor UI
This is a web-based UI specifically designed to work with the store-traffic-monitor-python reference implementation.   
The output of the Store-traffic-monitor reference implementation is read and displayed in real-time by the UI. For this reason the UI and reference implementation must be running in parallel. 

**To avoid any browser issues, the application should be started first, and then the UI.**

## Running the UI

Open new terminal and Go to UI directory present in this project directory:

```
cd UI
```

### Install the dependencies
```
sudo apt install composer
composer install
```
Run the following command on the terminal to open the UI.<br>

Chrome*:
```
google-chrome  --user-data-dir=$HOME/.config/google-chrome/store-traffic-monitor-python --new-window --allow-file-access-from-files --allow-file-access --allow-cross-origin-auth-prompt index.html
```

Firefox*:
```
firefox index.html
```
**_Note:_** For Firefox*, if the alerts list does not appear on the right side of the browser window, click anywhere on video progress bar to trigger a refresh.
