import sys
from urllib import unquote_plus

from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4.QtWebKit import *

class MyWebPage(QWebPage):
    formSubmitted = pyqtSignal(QUrl)

    def acceptNavigationRequest(self, frame, req, nav_type):
        if nav_type == QWebPage.NavigationTypeFormSubmitted:
            self.formSubmitted.emit(req.url())
        return super(MyWebPage, self).acceptNavigationRequest(frame, req, nav_type)

class Window(QWidget):
    def __init__(self, html):
        super(Window, self).__init__()
        self.setWindowFlags(Qt.FramelessWindowHint)
        view = QWebView(self)
        layout = QVBoxLayout(self)
        layout.addWidget(view)
        layout.setContentsMargins(0, 0, 0, 0)
        view.setPage(MyWebPage())
        view.setHtml(html)
        view.page().formSubmitted.connect(self.handleFormSubmitted)

    def handleFormSubmitted(self, url):
        self.close()
        elements = {}
        for key, value in url.encodedQueryItems():
            key = unquote_plus(bytes(key)).decode('utf8')
            value = unquote_plus(bytes(value)).decode('utf8')
            elements[key] = value
        # do stuff with elements...
        for item in elements.iteritems():
            print '"%s" = "%s"' % item
        qApp.quit()

# setup the html form
html = """
<form action="" method="get">
Like it?
<input type="radio" name="like" value="yes"/> Yes
<input type="radio" name="like" value="no" /> No
<br/><input type="text" name="text" value="" />
<input type="submit" name="submit" value="Send"/>
</form>
"""

def main():
    app = QApplication(sys.argv)

    window = Window(html)
    window.show()
    app.exec_()

if __name__ == "__main__":
    main()
