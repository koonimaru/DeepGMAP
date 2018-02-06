import Tkinter, tkFileDialog, tkMessageBox
import Tkinter, Tkconstants, tkFileDialog

class TkFileDialogExample(Tkinter.Frame):

  def __init__(self, root):

    Tkinter.Frame.__init__(self, root)

    # options for buttons
    button_opt = {'fill': Tkconstants.BOTH, 'padx': 5, 'pady': 5}

    # define buttons
    Tkinter.Button(self, text='askopenfile', command=self.askopenfile).pack(**button_opt)
    Tkinter.Button(self, text='askopenfilename', command=self.askopenfilename).pack(**button_opt)
    Tkinter.Button(self, text='asksaveasfile', command=self.asksaveasfile).pack(**button_opt)
    Tkinter.Button(self, text='asksaveasfilename', command=self.asksaveasfilename).pack(**button_opt)
    Tkinter.Button(self, text='askdirectory', command=self.askdirectory).pack(**button_opt)

    # define options for opening or saving a file
    self.file_opt = options = {}
    options['defaultextension'] = '.txt'
    options['filetypes'] = [('all files', '.*'), ('text files', '.txt')]
    options['initialdir'] = '/home/'
    options['initialfile'] = 'myfile.txt'
    options['parent'] = root
    options['title'] = 'This is a title'

    # This is only available on the Macintosh, and only when Navigation Services are installed.
    #options['message'] = 'message'

    # if you use the multiple file version of the module functions this option is set automatically.
    #options['multiple'] = 1

    # defining options for opening a directory
    self.dir_opt = options = {}
    options['initialdir'] = '/home/'
    options['mustexist'] = False
    options['parent'] = root
    options['title'] = 'This is a title'

  def askopenfile(self):

    """Returns an opened file in read mode."""

    return tkFileDialog.askopenfile(mode='r', **self.file_opt)

  def askopenfilename(self):

    """Returns an opened file in read mode.
    This time the dialog just returns a filename and the file is opened by your own code.
    """

    # get filename
    filename = tkFileDialog.askopenfilename(**self.file_opt)

    # open file on your own
    if filename:
      return open(filename, 'r')

  def asksaveasfile(self):

    """Returns an opened file in write mode."""

    return tkFileDialog.asksaveasfile(mode='w', **self.file_opt)

  def asksaveasfilename(self):

    """Returns an opened file in write mode.
    This time the dialog just returns a filename and the file is opened by your own code.
    """

    # get filename
    filename = tkFileDialog.asksaveasfilename(**self.file_opt)

    # open file on your own
    if filename:
      return open(filename, 'w')

  def askdirectory(self):

    """Returns a selected directoryname."""

    return tkFileDialog.askdirectory(**self.dir_opt)

if __name__=='__main__':
  root = Tkinter.Tk()
  TkFileDialogExample(root).pack()
  root.mainloop()
  
"""  
if __name__ == '__main__':
    form = Tkinter.Tk()

    getFld = Tkinter.IntVar()

    form.wm_title('DeepShark GUI')

    stepOne = Tkinter.LabelFrame(form, text=" 1. Enter input file direction: ")
    stepOne.grid(row=0, columnspan=7, sticky='W', \
                 padx=5, pady=5, ipadx=5, ipady=5)

    helpLf = Tkinter.LabelFrame(form, text=" Quick Help ")
    helpLf.grid(row=0, column=9, columnspan=2, rowspan=8, \
                sticky='NS', padx=5, pady=5)
    helpLbl = Tkinter.Label(helpLf, text="Help will come - ask for it.")
    helpLbl.grid(row=0)

    stepTwo = Tkinter.LabelFrame(form, text=" 2. Enter Table Details: ")
    stepTwo.grid(row=2, columnspan=7, sticky='W', \
                 padx=5, pady=5, ipadx=5, ipady=5)

    stepThree = Tkinter.LabelFrame(form, text=" 3. Configure: ")
    stepThree.grid(row=3, columnspan=7, sticky='W', \
                   padx=5, pady=5, ipadx=5, ipady=5)

    inFileLbl = Tkinter.Label(stepOne, text="Select the File:")
    inFileLbl.grid(row=0, column=0, sticky='E', padx=5, pady=2)

    inFileTxt = Tkinter.Entry(stepOne)
    inFileTxt.grid(row=0, column=1, columnspan=7, sticky="WE", pady=3)
    inputdir=''
    def loadinputfiles(inputdir): 
        filename = tkFileDialog.askdirectory()
        if filename: 
            filename.set(filename)
            
    inFileBtn = Tkinter.Button(stepOne, text = "Browse", command = loadinputfiles(inputdir))
    inFileBtn.grid(row=0, column=8, sticky='W', padx=5, pady=2)


    outFileLbl = Tkinter.Label(stepOne, text="Save File to:")
    outFileLbl.grid(row=1, column=0, sticky='E', padx=5, pady=2)

    outFileTxt = Tkinter.Entry(stepOne)
    outFileTxt.grid(row=1, column=1, columnspan=7, sticky="WE", pady=2)

    outFileBtn = Tkinter.Button(stepOne, text="Browse ...")
    outFileBtn.grid(row=1, column=8, sticky='W', padx=5, pady=2)

    inEncLbl = Tkinter.Label(stepOne, text="Input File Encoding:")
    inEncLbl.grid(row=2, column=0, sticky='E', padx=5, pady=2)

    inEncTxt = Tkinter.Entry(stepOne)
    inEncTxt.grid(row=2, column=1, sticky='E', pady=2)

    outEncLbl = Tkinter.Label(stepOne, text="Output File Encoding:")
    outEncLbl.grid(row=2, column=5, padx=5, pady=2)

    outEncTxt = Tkinter.Entry(stepOne)
    outEncTxt.grid(row=2, column=7, pady=2)

    outTblLbl = Tkinter.Label(stepTwo, \
          text="Enter the name of the table to be used in the statements:")
    outTblLbl.grid(row=3, column=0, sticky='W', padx=5, pady=2)

    outTblTxt = Tkinter.Entry(stepTwo)
    outTblTxt.grid(row=3, column=1, columnspan=3, pady=2, sticky='WE')

    fldLbl = Tkinter.Label(stepTwo, \
                           text="Enter the field (column) names of the table:")
    fldLbl.grid(row=4, column=0, padx=5, pady=2, sticky='W')

    getFldChk = Tkinter.Checkbutton(stepTwo, \
                           text="Get fields automatically from input file",\
                           onvalue=1, offvalue=0)
    getFldChk.grid(row=4, column=1, columnspan=3, pady=2, sticky='WE')

    fldRowTxt = Tkinter.Entry(stepTwo)
    fldRowTxt.grid(row=5, columnspan=5, padx=5, pady=2, sticky='WE')

    transChk = Tkinter.Checkbutton(stepThree, \
               text="Enable Transaction", onvalue=1, offvalue=0)
    transChk.grid(row=6, sticky='W', padx=5, pady=2)

    transRwLbl = Tkinter.Label(stepThree, \
                 text=" => Specify number of rows per transaction:")
    transRwLbl.grid(row=6, column=2, columnspan=2, \
                    sticky='W', padx=5, pady=2)

    transRwTxt = Tkinter.Entry(stepThree)
    transRwTxt.grid(row=6, column=4, sticky='WE')

    form.mainloop()"""