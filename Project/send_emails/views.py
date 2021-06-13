from django.shortcuts import render
from django.views import View
from django.http import HttpResponseRedirect
from django.core.mail import EmailMessage
from django.conf import settings
from .forms import SendEmail
from django.contrib import messages
import time
# Create your views here.


class EmailAttchementView(View):
    form_class = SendEmail
    templates_name = 'send_email.html'

    def get(self,request,*args, **kwargs):
        form = self.form_class()
        return render(request,self.templates_name,{'email_form':form})
    
    def post(self,request,*args, **kwargs):
        form = self.form_class(request.POST,request.FILES)
        if form.is_valid():
            subject = form.cleaned_data['subject']
            message = form.cleaned_data['message']
            email = form.cleaned_data['email']
            files = request.FILES.getlist('attach')
            try:
                mail = EmailMessage(subject,message,settings.EMAIL_HOST_USER,[email])
                for f in files:
                    mail.attach(f.name,f.read(),f.content_type)
                mail.send()
                time.sleep(10)
                return render(request,self.templates_name,{'email_form':form,'error_message':'Sent email to %s'%email})
                
            except:
                return render(request,self.templates_name,{'email_form':form,'error_message':'Eighter the attachement is too big or invalid'})
        return render(request,self.templates_name,{'email_form':form,'error_message':"Unable to send email. Please try again later..!!"})