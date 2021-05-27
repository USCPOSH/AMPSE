

# pip install xlrd
# pip install ExcelWriter
import pandas as pd
import numpy as np
import pandas.io.formats.excel





def readexcel(fexcel):
	# Read User Template file by pandas Xlsx writer:
	df = pd.read_excel(fexcel, sheetname='Sheet1')
	data=df['Values']
	
	address=data.iloc[0:2].values
	inputs=[]
	inputs.append(data.iloc[2:4].values)
	inputs.append(data.iloc[5:15].values)
	inputs.append(data.iloc[16:29].values)
	return [address,inputs]

def toexcel(df,fexcel):
	# Create a Pandas Excel writer using XlsxWriter as the engine.
	
	writer = pd.ExcelWriter(fexcel, engine='xlsxwriter')
	#pd.formats.format.header_style = None
	pd.io.formats.excel.header_style = None


	# Convert the dataframe to an XlsxWriter Excel object.
	df.to_excel(writer,sheet_name='Sheet1')
	
	
	# Get the xlsxwriter workbook and worksheet objects.
	workbook  = writer.book
	worksheet = writer.sheets['Sheet1']

	# Add some cell formats.
	format1 = workbook.add_format({'num_format': '#,##0.00'})
	format2 = workbook.add_format({'font_size':28,'bold':True})#bold
	format3 = workbook.add_format({'font_size': 24,'num_format': '#,##0.00'})#big font size
	formatred = workbook.add_format({'bg_color': '#FFC7CE','font_color': '#9C0006'})#red
	formatgreen = workbook.add_format({'bg_color': '#C6EFCE','font_color': '#006100'})#green

	# Note: It isn't possible to format any cells that already have a format such
	# as the index or headers or any cells that contain dates or datetimes.

	# Set the column width and format.
	worksheet.set_column('B:F', 50, format3)
	worksheet.set_default_row(30)
	worksheet.set_row(0,None,format2)


	worksheet.conditional_format('F2:F11', {'type': 'text','criteria':'containing','value': 'failed','format': formatred})
	worksheet.conditional_format('F2:F11', {'type': 'text','criteria':'containing','value': 'passed','format': formatgreen})
	#worksheet.conditional_format('E2:E11', {'type': '3_color_scale'})

	# Close the Pandas Excel writer and output the Excel file.
	writer.save()
def toexcel_all(df,df_static,df_dynamic,fexcel):
	# Create a Pandas Excel writer using XlsxWriter as the engine.

	writer = pd.ExcelWriter(fexcel, engine='xlsxwriter')
	#pd.formats.format.header_style = None
	pd.io.formats.excel.header_style = None


	# Convert the dataframe to an XlsxWriter Excel object.
	df.to_excel(writer,sheet_name='Sheet1')
	df_static.to_excel(writer,sheet_name='Sheet2')    
	df_dynamic.to_excel(writer,sheet_name='Sheet3')	


	workbook  = writer.book	
	# ---------------------------------------------    
	# Formating the first sheet i.e. Summary page
	# ---------------------------------------------
	worksheet = writer.sheets['Sheet1']

	# Add some cell formats.
	format1 = workbook.add_format({'num_format': '#,##0.00'})
	format2 = workbook.add_format({'font_size':28,'bold':True})#bold
	format3 = workbook.add_format({'font_size': 24,'num_format': '#,##0.00'})#big font size
	formatred = workbook.add_format({'bg_color': '#FFC7CE','font_color': '#9C0006'})#red
	formatgreen = workbook.add_format({'bg_color': '#C6EFCE','font_color': '#006100'})#green

	# Set the column width and format.
	worksheet.set_column('B:F', 50, format3)
	worksheet.set_default_row(30)
	worksheet.set_row(0,None,format2)
	worksheet.conditional_format('F2:F11', {'type': 'text','criteria':'containing','value': 'failed','format': formatred})
	worksheet.conditional_format('F2:F11', {'type': 'text','criteria':'containing','value': 'passed','format': formatgreen})

	# ---------------------------------------------
	# Formating the Second sheet i.e. INL & DNL page:
	# ---------------------------------------------    
	worksheet = writer.sheets['Sheet2']    
	# Create a new chart object. In this case an embedded chart.
	chart1 = workbook.add_chart({'type': 'line'})
	lbit=len(df_static)
	# Configure the first series.
	chart1.add_series({
	    'name':       ['Sheet2', 0, 1],
	    'categories': ['Sheet2', 1, 0, lbit, 0],
	    'values':     ['Sheet2', 1, 1, lbit, 1],
	})

	# Configure second series. Note use of alternative syntax to define ranges.
	chart1.add_series({
	    'name':       ['Sheet2', 0, 2],
	    'categories': ['Sheet2', 1, 0, lbit, 0],
	    'values':     ['Sheet2', 1, 2, lbit, 2],
	})

	# Add a chart title and some axis labels.
	chart1.set_title ({'name': 'ADC Static Analysis'})
	chart1.set_x_axis({'name': 'Digital Code','name_font': {'size': 14, 'bold': True}})
	chart1.set_y_axis({'name': '(LSB)','name_font': {'size': 14, 'bold': True}})
	chart1.set_legend({'font': {'size': 14, 'bold': True}})

	chart1.set_style(10)

	# Insert the chart into the worksheet (with an offset).
	worksheet.insert_chart('F2', chart1, {'x_offset': 25, 'y_offset': 10})

	# ---------------------------------------------
	# Formating the Third sheet i.e. FFT  page:
	# ---------------------------------------------    
	worksheet = writer.sheets['Sheet3']    
	# Create a new chart object. In this case an embedded chart.
	chart2 = workbook.add_chart({'type': 'line'})
	    
	lsin=len(df_dynamic)
	# Configure the first series.
	chart2.add_series({'name':'Signal','categories':['Sheet3', 1, 11, lsin, 11],'values':['Sheet3', 1, 1, lsin, 1]})
	chart2.add_series({'name':'H2'    ,'categories':['Sheet3', 1, 11, lsin, 11],'values':['Sheet3', 1, 3, lsin, 3]})
	chart2.add_series({'name':'H3'    ,'categories':['Sheet3', 1, 11, lsin, 11],'values':['Sheet3', 1, 4, lsin, 4]})
	chart2.add_series({'name':'H4'    ,'categories':['Sheet3', 1, 11, lsin, 11],'values':['Sheet3', 1, 5, lsin, 5]})
	chart2.add_series({'name':'H5'    ,'categories':['Sheet3', 1, 11, lsin, 11],'values':['Sheet3', 1, 6, lsin, 6]})
	chart2.add_series({'name':'H6'    ,'categories':['Sheet3', 1, 11, lsin, 11],'values':['Sheet3', 1, 7, lsin, 7]})
	chart2.add_series({'name':'H7'    ,'categories':['Sheet3', 1, 11, lsin, 11],'values':['Sheet3', 1, 8, lsin, 8]})
	chart2.add_series({'name':'H8'    ,'categories':['Sheet3', 1, 11, lsin, 11],'values':['Sheet3', 1, 9, lsin, 9]})    
	chart2.add_series({'name':'Noise' ,'categories':['Sheet3', 1, 11, lsin, 11],'values':['Sheet3', 1, 10, lsin, 10]})        
	# Add a chart title and some axis labels.
	chart2.set_title ({'name': 'ADC Dynamic Analysis'})
	chart2.set_x_axis({'name': 'Frequency (MHz)','name_font': {'size': 14, 'bold': True},'min':1e-9,'max':10})
	chart2.set_y_axis({'name': '(dB)','name_font': {'size': 14, 'bold': True},'log_base': 10})
	chart2.set_legend({'font': {'size': 14, 'bold': True}})
	chart2.set_size({'width': 720, 'height': 576})

	chart2.set_style(10)

	# Insert the chart into the worksheet (with an offset).
	worksheet.insert_chart('B2', chart2, {'x_offset': 25, 'y_offset': 10})






	writer.save()
    
    
def static_to_excel(fexcel,statics):
	# Create a Pandas Excel writer using XlsxWriter as the engine.
	writer = pd.ExcelWriter(fexcel, engine='xlsxwriter')
	# Convert the dataframe to an XlsxWriter Excel object.
	statics.to_excel(writer,sheet_name='Sheet1')
	
	# Get the xlsxwriter workbook and worksheet objects.
	workbook  = writer.book
	worksheet = writer.sheets['Sheet1']
	

	# Chart starts here:
	chart = workbook.add_chart({'type': 'line'})
	# Configure the series of the chart from the dataframe data.
	chart.add_series({'values':     '=Sheet1!$B$2:$B$47'})

	# Configure the chart axes.
	chart.set_y_axis({'major_gridlines': {'visible': False}})
	
	writer.save()


if __name__ == "__main__":
	print(readexcel('UserTemplate.xlsx'))

