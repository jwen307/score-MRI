# REQUIRED IMPORTS
import os
import base64

from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import *



# file_type should be appropriate extension, e.g. png, jpg, gif, etc.
def send_mail(subject, body_text, attachments=[]):
    """
    Sends and email with custom content.
    Args:
        subject: (str) The subject to be used for the email.
        body_text: (str) The plaintext body content for the email.
        attachments: (list) A list of attachments to be added to the email.
                     Each element should be a dict with key file_name, file_path, and file_type.
    """
    # ARGUMENTS FOR MAIL HELPER CLASS - Use 'help(Mail)' to see all arguments, functions, and attributes
    # of the Mail class
    FROM_EMAIL = Email('schnitergroup@gmail.com') # DO NOT CHANGE THIS
    TO_EMAIL = To('wen.254@osu.edu') # MAKE THIS YOUR EMAIL
    SUBJECT = subject
    CONTENT = Content("text/plain", body_text)

    # SENDGRID MAIL HELPER CLASS - SEE DOCS FOR HOW TO CUSOTMIZE THIS
    message = Mail(FROM_EMAIL, TO_EMAIL, SUBJECT, CONTENT)

    # ADD ATTACHMENTS TO EMAIL IF THERE ARE ANY
    if len(attachments) > 0:
        """
        Each attachment should be represented by a dict of the following form:
        {
            'file_name': 'FILE NAME AS IT WILL APPEAR IN THE EMAIL',
            'file_path': 'THE ACTUAL PATH TO THE FILE, ABSOLUTE OR RELATIVE',
            'file_type': 'THE FILE TYPE - SHOULD BE THE EXTENSION WITHOUT THE LEADING PERIOD',
        }
        
        See SendGrid's documentation for supported file types.
        """
        for attachment in attachments:
            try:
                with open(attachment["file_path"], 'rb') as f:
                    data = f.read()
                    f.close()
                encoded_file = base64.b64encode(data).decode()

                attachedFile = Attachment(
                    FileContent(encoded_file),
                    FileName(attachment["file_name"]),
                    FileType(attachment["file_type"]),
                    Disposition('attachment')
                )
                message.add_attachment(attachedFile)
            except Exception as e:
                print("Unable to add attachment, got the following error:")
                print(e)

    try:
        SENDGRID_API_KEY = os.environ.get('SENDGRIDAPI')
        if SENDGRID_API_KEY is None:
            raise ValueError('The SendGrid API key has not been set as an environment variable')
        
        
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        sg.send(message)
    except Exception as e:
        print(e.message)


# EXAMPLE CODE
def example():
    SUBJECT = "EXAMPLE EMAIL"
    BODY = "This is an example body."
    ATTACHMENTS = [
        {
            "file_name": "my_gif.gif",
            "file_path": "/path/to/my/gif/my_gif.gif",
            "file_type": "gif",
        },
        {
            "file_name": "my_png.png",
            "file_path": "/path/to/my/png/my_png.png",
            "file_type": "png",
        },
    ]

    send_mail(SUBJECT, BODY, attachments=ATTACHMENTS)
