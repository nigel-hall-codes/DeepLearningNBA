import os


class QOLTools:

    def clear_found_players_dir(self):
        for file in os.listdir(r"D:\PycharmProjects\NBACV\found_players"):
            os.remove(r"D:\PycharmProjects\NBACV\found_players/{}".format(file))


tools = QOLTools()
tools.clear_found_players_dir()