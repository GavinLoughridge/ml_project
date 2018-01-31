# ml_project

this is an experiment with machine learning.

in it's current form it recognizes handwritten digits from pictures, which you can try out on heroku here:
digitreader.herokuapp.com

the neral network was made using tensorflow and the MNIST dataset.

the app recives a picture, then goes through a preprocessing pipeline to enahance, center, and resize the image
so that it better matches the training data from MNIST. then the resultant array is classified by the neral network 
and the networks 'guess' along with a copy of the processed image is sent back to the web app.

I used Vue to render the single page app, which made it easy to update the view as the app recives data from users and 
the server.
