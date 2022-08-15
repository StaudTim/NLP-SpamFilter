import pandas as pd
from imap_tools import MailBox


def get_data_from_mail():
    user = input("E-Mail: ")
    password = input("password: ")
    imap = input("imap e.g. imap.gmail.com: ")

    folders = ['INBOX', 'Spam']
    text = []
    label = []
    for folder in folders:
        mb = MailBox(imap).login(user, password, initial_folder=folder)
        messages = mb.fetch(mark_seen=False, bulk=True)

        for msg in messages:
            text.append(msg.subject)
            if folder == 'INBOX':
                label.append('ham')
            else:
                label.append('spam')

        mb.logout()

    data = {'label': label,
            'text': text}
    df_data = pd.DataFrame(data)
    df_data.to_csv("dataset.csv")


def main():
    get_data_from_mail()


if __name__ == "__main__":
    main()
