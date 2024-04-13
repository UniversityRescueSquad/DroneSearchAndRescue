import panel as pn

pn.extension()

class VideoMission:
    def __init__(self, path):
        self.path = path
        

def app():
    return "## Hello world!"

pn.serve(
    {'/': app},
    port=8888,
)
